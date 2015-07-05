#!/usr/bin/python
import sys, getopt, random, numpy, os
import configuration, neuralnet, translate, shiftreducer

usage = "\n\nNAME\n\n\n\toracles - train parser oracles\n\n\nSYNOPSIS\n\n\n\t./oracles.py [OPTIONS]... <FILE:VEC> <FILE|DIR:PAR> <ARRAY>\n\n\nDESCRIPTION\n\n\n\tTrain shift reduce dependency parser oracles. Use neural network language model. Learn phrase vectors automatically. Print to stdout unless write file specified.\n\n\tMandatory arguments to long options are mandatory for short options too.\n\n\nOPTIONS\n\n\n\t-e, --embedding=INT[200]\n\t\taccept size of vector space embedding\n\n\t-h, --help\n\t\tdisplay this help and exit\n\n\t-i, --iterations=INT[1]\n\t\taccept number of iterations to train for\n\n\t-p, --phraser\n\t\ttrain phraser, instead of shift reducer\n\n\t-s, --sentence=INT[100]\n\t\taccept maximum sentence length\n\n\t-v, --verbose\n\t\tdisplay information for debugging\n\n\t-w, --write=FILE\n\t\taccept file to write trained oracle to\n\n\nARGUMENTS\n\n\n\tFILE:VEC\n\t\taccept file containing word vectors\n\n\tFILE:PAR\n\t\taccept file containing Standford typed dependencies\n\n\tDIR:PAR\n\t\taccept directory containing Standford typed dependencies\n\n\tARRAY\n\t\taccept dependency labels for training\n\n\nAUTHOR\n\n\n\tWritten by Amitrajit Sarkar.\n\n\nREPORTING BUGS\n\n\n\tReport bugs to <aaiijmrtt@gmail.com>.\n\n\nCOPYRIGHT\n\n\nThe MIT License (MIT)\n\nCopyright (c) 2015 Amitrajit Sarkar\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"

def error(description):

	print description
	sys.exit()

def arguments(string):

	labels = None
	vecnamein = None
	namein = None
	dirnamein = None
	write = None
	oraclenamein = None

	try:
		opts, args = getopt.getopt(string, "hvps:e:w:o:i:", ["help=", "verbose", "phraser", "sentence=", "embedding=", "write=", "oracle=", "iterations="])
		for opt, arg in opts:
			if opt in ("-h", "--help"):
				error(usage)
			elif opt in ("-v", "--verbose"):
				configuration.verbose = True
			elif opt in ("-p", "--phraser"):
				configuration.phraser = True
			elif opt in ("-s", "--sentence"):
				configuration.maxlength = int(arg)
			elif opt in ("-e", "--embedding"):
				configuration.embeddingsize = int(arg)
			elif opt in ("-w", "--write"):
				write = arg
			elif opt in ("-o", "--oracle"):
				oraclenamein = arg
			elif opt in ("-i", "--iterations"):
				configuration.iterations = int(arg)

	except Exception:
		error(usage)

	if len(args) < 2:
		error(usage)

	vecnamein = args[0]
	if os.path.isdir(args[1]):
		dirnamein = args[1]
	else:
		namein = args[1]
	labels = args[2:]

	if configuration.verbose:
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT sentence: %d]" %configuration.maxlength
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT embedding: %s]" %configuration.embeddingsize
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT phraser: %s]" %configuration.phraser
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT write: %s]" %write
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT oraclefile: %s]" %oraclenamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT vectorfile: %s]" %vecnamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsefile: %s]" %namein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsedirectory: %s]" %dirnamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT labels: %s]" %labels

	return write, oraclenamein, vecnamein, namein, dirnamein, labels

def trainphraser(oracle, labels, dependencyfile, vectorspace):

	droppedcount = 0
	inputs = list()
	outputs = list()

	while dependencyfile.tell() != os.fstat(dependencyfile.fileno()).st_size:
		readline = dependencyfile.readline()
		match = configuration.parse.match(readline)

		if match:
			split1 = match.group(2).rfind('-')
			word1 = match.group(2)[0: split1]
			index1 = int(match.group(2)[split1 + 1:])
			split2 = match.group(3).rfind('-')
			word2 = match.group(3)[0: split2]
			index2 = int(match.group(3)[split2 + 1:])

			if word1.lower() in vectorspace and word1 != 'ROOT'and word2.lower() in vectorspace and word2 != 'ROOT':
				inputs.append(numpy.concatenate([vectorspace[word1.lower()], vectorspace[word2.lower()]]))
				label = len(labels)
				for i in range(len(labels)):
					if labels[i] == match.group(1):
						label = i
				label += (len(labels) + 1) * shiftreducer.REDUCERIGHT if index1 < index2 else (len(labels) + 1) * shiftreducer.REDUCELEFT
				output = numpy.zeros((2 * (len(labels) + 1) + 1, 1), dtype = float)
				output[label][0] = 1.0
				outputs.append(output)

			else:
				droppedcount += 1

	altogether = list(zip(inputs, outputs))
	random.shuffle(altogether)
	inputs, outputs = zip(*altogether)
	traininputs = inputs[0: int(0.80 * len(inputs))]
	trainoutputs = outputs[0: int(0.80 * len(outputs))]
	testinputs = inputs[int(0.80 * len(inputs)): ]
	testoutputs = outputs[int(0.80 * len(outputs)): ]

	for i in range(configuration.iterations):
		oracle['weights'], oracle['biases'] = neuralnet.train(traininputs, trainoutputs, oracle['weights'], oracle['biases'])
		error = neuralnet.testclassifier(testinputs, testoutputs, oracle['weights'], oracle['biases'])
		if configuration.verbose:
			print "[DEBUG: ORACLE MISSED ON %d DEPENDENCIES IN FILE %s]" %(droppedcount, dependencyfile.name)
			print "[DEBUG: ORACLE TRAINED ON %d DEPENDENCIES IN FILE %s]" %(len(traininputs), dependencyfile.name)
			print "[DEBUG: ORACLE TESTED ON %d DEPENDENCIES IN FILE %s]" %(len(testinputs), dependencyfile.name)
			print "[DEBUG: ORACLE ERROR %f]" %error

	return oracle, len(traininputs), error

def trainshiftreducer(oracle, labels, dependencyfile, vectorspace):

	trainingcount = 0
	droppedcount = 0
	errorcount = 0

	while dependencyfile.tell() != os.fstat(dependencyfile.fileno()).st_size:
		endoftreeflag = False
		validtreeflag = True
		inputs = [None for i in range(configuration.maxlength)]
		outputs = list()
		wordcount = 0

		while not endoftreeflag:
			readline = dependencyfile.readline()
			match = configuration.parse.match(readline)

			if match:
				validlabelflag = False
				split1 = match.group(2).rfind('-')
				word1 = match.group(2)[0: split1]
				index1 = int(match.group(2)[split1 + 1:])
				split2 = match.group(3).rfind('-')
				word2 = match.group(3)[0: split2]
				index2 = int(match.group(3)[split2 + 1:])

				if word1.lower() in vectorspace and word1 != 'ROOT':
					inputs[index1] = vectorspace[word1.lower()]
				elif word1.lower() not in vectorspace and word1 != 'ROOT':
					validtreeflag = False

				if word2.lower() in vectorspace and word2 != 'ROOT':
					inputs[index2] = vectorspace[word2.lower()]
				elif word2.lower() not in vectorspace and word2 != 'ROOT':
					validtreeflag = False

				label = len(labels)
				for i in range(len(labels)):
					if labels[i] == match.group(1):
						label = i

				label += (len(labels) + 1) * shiftreducer.REDUCERIGHT if index1 < index2 else (len(labels) + 1) * shiftreducer.REDUCELEFT
				outputs.append((index1, index2, label))
				wordcount = max(wordcount, index1, index2)

			else:
				endoftreeflag = True

		if validtreeflag:
			inputs[0] = vectorspace['</s>']
			inputs = [numpy.reshape(inp, (configuration.embeddingsize, 1)) for inp in inputs if inp is not None]
			if len(inputs) != wordcount + 1:
				droppedcount += 1
				continue
			try:
				oracle = shiftreducer.trainoracles(inputs, outputs, oracle, len(labels))
				trainingcount += 1
			except Exception, e:
				errorcount += 1
		else:
			droppedcount += 1

	if configuration.verbose:
		print "[DEBUG: ORACLE MISSED ON %d SENTENCES IN FILE %s]" %(droppedcount, dependencyfile.name)
		print "[DEBUG: ORACLE DROPPED ON %d SENTENCES IN FILE %s]" %(errorcount, dependencyfile.name)
		print "[DEBUG: ORACLE TRAINED ON %d SENTENCES IN FILE %s]" %(trainingcount, dependencyfile.name)

	return oracle, trainingcount

def readoracle(oraclenamein, labels = None):

	if oraclenamein is None:
		weights = [numpy.random.rand(configuration.embeddingsize, configuration.contextnumber * configuration.embeddingsize), numpy.random.rand(2 * labels + 3, configuration.embeddingsize)]
		biases = [numpy.random.rand(configuration.embeddingsize, 1), numpy.random.rand(2 * labels + 3, 1)]
		if configuration.verbose:
			print "[DEBUG: ORACLE INITIALIZED RANDOMLY]"

	else:
		with numpy.load(oraclenamein) as oraclein:
			weights = [oraclein['weightsin'], oraclein['weightsout']]
			biases = [oraclein['biasesin'], oraclein['biasesout']]
		if configuration.verbose:
			print "[DEBUG: ORACLE INITIALIZED FROM FILE %s]" %oraclenamein

	return {'weights': weights, 'biases': biases}

def writeoracle(oraclenameout):

	if oraclenameout is None:
		print "weightsin", oracle['weights'][0], "weightsout", oracle['weights'][1], "biasesin", oracle['biases'][0], "biasesout", oracle['biases'][1]
	else:
		numpy.savez(oraclenameout, weightsin = oracle['weights'][0], weightsout = oracle['weights'][1], biasesin = oracle['biases'][0], biasesout = oracle['biases'][1])
		if configuration.verbose:
			print "[DEBUG: ORACLE WRITTEN TO FILE %s]" %oraclenameout

if __name__ == '__main__':

	oraclenameout, oraclenamein, vecnamein, namein, dirnamein, labels = arguments(sys.argv[1:])
	vectorspace, vectorsize = translate.vectorspace(vecnamein)
	oracle = readoracle(oraclenamein, len(labels))
	if vectorsize != configuration.embeddingsize:
		error("[ERROR: EMBEDDING SIZE MISMATCH]")

	trainingcount = 0

	if dirnamein is None:
		if namein is None:
			error(usage)
		with open(namein, "r") as filein:
			if configuration.phraser:
				oracle, count, error = trainphraser(oracle, labels, filein, vectorspace)
			else:
				oracle, count = trainshiftreducer(oracle, labels, filein, vectorspace)
			trainingcount += count

	else:
		for root, dirs, files in os.walk(dirnamein, topdown = False):
			for name in files:
				with open(os.path.join(root, name), "r") as filein:
					if configuration.phraser:
						oracle, count, error = trainphraser(oracle, labels, filein, vectorspace)
					else:
						oracle, count = trainshiftreducer(oracle, labels, filein, vectorspace)
					trainingcount += count

	if configuration.verbose:
		print "[DEBUG: ORACLE TRAINED ON %d PARSED INSTANCES]" %trainingcount
	writeoracle(oraclenameout)
