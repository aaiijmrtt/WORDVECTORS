#!/usr/bin/python
import sys, getopt, random, numpy, os
import configuration, neuralnet, translate, shiftreducer, dependencies, graphparser

usage = "\n\nNAME\n\n\n\toracles - train parser oracles\n\n\nSYNOPSIS\n\n\n\t./oracles.py [OPTIONS]... <FILE:VEC> <FILE|DIR:PAR> <ARRAY>\n\n\nDESCRIPTION\n\n\n\tTrain shift reduce dependency parser oracles. Use neural network language model. Learn phrase vectors automatically. Print to stdout unless write file specified.\n\n\tMandatory arguments to long options are mandatory for short options too.\n\n\nOPTIONS\n\n\n\t-e, --embedding=INT[200]\n\t\taccept size of vector space embedding\n\n\t-h, --help\n\t\tdisplay this help and exit\n\n\t-i, --iterations=INT[1]\n\t\taccept number of iterations to train for\n\n\t-o, --oracle=FILE\n\t\taccept file containing oracle to be trained\n\n\t-s, --sentence=INT[100]\n\t\taccept maximum sentence length\n\n\t-t, --type=STRING[Stanford]\n\t\taccept dependency file type\n\n\t-v, --verbose\n\t\tdisplay information for debugging\n\n\t-w, --write=FILE\n\t\taccept file to write trained oracle to\n\n\t-x, --cross=FILE\n\t\taccept file containing translator for crosslingual training\n\n\nARGUMENTS\n\n\n\tFILE:VEC\n\t\taccept file containing word vectors\n\n\tFILE:PAR\n\t\taccept file containing Standford typed dependencies\n\n\tDIR:PAR\n\t\taccept directory containing Standford typed dependencies\n\n\tARRAY\n\t\taccept dependency labels for training\n\n\nAUTHOR\n\n\n\tWritten by Amitrajit Sarkar.\n\n\nREPORTING BUGS\n\n\n\tReport bugs to <aaiijmrtt@gmail.com>.\n\n\nCOPYRIGHT\n\n\nThe MIT License (MIT)\n\nCopyright (c) 2015 Amitrajit Sarkar\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"

def arguments(string):

	labels = None
	vecnamein = None
	namein = None
	dirnamein = None
	write = None
	oraclenamein = None
	translator = None

	try:
		opts, args = getopt.getopt(string, "hvs:e:w:o:i:t:x:f", ["help=", "verbose", "sentence=", "embedding=", "write=", "oracle=", "iterations=", "type=", "cross=", "flag"])
		for opt, arg in opts:
			if opt in ("-h", "--help"):
				configuration.error(usage)
			elif opt in ("-v", "--verbose"):
				configuration.verbose = True
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
			elif opt in ("-t", "--type"):
				configuration.filetype = arg
			elif opt in ("-x", "--cross"):
				translator = arg
			elif opt in ("-f", "--flag"):
				configuration.train = False

	except Exception:
		configuration.error(usage)

	if len(args) < 2:
		configuration.error(usage)

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
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT filetype: %s]" %configuration.filetype
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT train: %s]" %configuration.train
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT translator: %s]" %translator
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT write: %s]" %write
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT oraclefile: %s]" %oraclenamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT vectorfile: %s]" %vecnamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsedfile: %s]" %namein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsedirectory: %s]" %dirnamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT labels: %s]" %labels

	return write, oraclenamein, translator, vecnamein, namein, dirnamein, labels

def trainshiftreducer(oracle, labels, sentences, vectors):

	trainingcount = 0
	testingcount = 0
	droppedcount = 0
	errorcount = 0
	incorrectcount = 0

	for i in range(int(0.80 * len(sentences))):
		validtree = True
		for j in range(1, len(vectors[i])):
			if vectors[i][j] is None:
				validtree = False
				break

		if validtree:
			inputs = vectors[i]
			outputs = list()
			for j in range(len(sentences[i])):
				label = len(labels)
				for k in range(len(labels)):
					if labels[k] == dependency[2]:
						label = k
				label += (len(labels) + 1) * shiftreducer.REDUCERIGHT if sentences[i][j][0] < sentences[i][j][1] else (len(labels) + 1) * shiftreducer.REDUCELEFT
				outputs.append((sentences[i][j][0], sentences[i][j][1], label))

			try:
				oracle = shiftreducer.trainoracle(inputs, outputs, oracle, len(labels))
				trainingcount += 1

			except Exception, e:
				print "[EXCEPTION: %s IN TRAINING SENTENCE %d]" %(e, i)
				errorcount += 1

		else:
			droppedcount += 1

		if configuration.verbose:
			sys.stdout.write("[TRAINING PROGRESS: %f]\r" %(float(i) / len(sentences)))
			sys.stdout.flush()

	for i in range(int(0.80 * len(sentences)), len(sentences)):
		validtree = True
		for j in range(1, len(vectors[i])):
			if vectors[i][j] is None:
				validtree = False
				break

		if validtree:
			outputs = shiftreducer.shiftreduce(vectors[i][1: ], oracle)
			for j in range(len(sentences[i])):
				label = len(labels)
				for k in range(len(labels)):
					if labels[k] == dependency[2]:
						label = k
				label += (len(labels) + 1) * shiftreducer.REDUCERIGHT if sentences[i][j][0] < sentences[i][j][1] else (len(labels) + 1) * shiftreducer.REDUCELEFT
				if (sentences[i][j][0], sentences[i][j][1], label) not in outputs:
					incorrectcount += 1
				testingcount += 1

		else:
			droppedcount += 1

	if configuration.verbose:
		print "[DEBUG: ORACLE MISSED ON %d SENTENCES]" %droppedcount
		print "[DEBUG: ORACLE DROPPED ON %d SENTENCES]" %errorcount
		print "[DEBUG: ORACLE TRAINED ON %d SENTENCES]" %trainingcount
		print "[DEBUG: ORACLE TESTED ON %d DEPENDENCIES]" %testingcount
		print "[DEBUG: ORACLE ERROR: %f]" %(float(incorrectcount) / testingcount)

	return oracle, trainingcount

def readoracle(oraclenamein, labels = None):

	if oraclenamein is None:
		weights = [numpy.random.rand(configuration.embeddingsize, configuration.contextnumber * configuration.embeddingsize), numpy.random.rand(2 * labels + 3, configuration.embeddingsize + 2 * labels + 3)]
		biases = [numpy.random.rand(configuration.embeddingsize, 1), numpy.random.rand(2 * labels + 3, 1)]
		if configuration.verbose:
			print "[DEBUG: ORACLE INITIALIZED RANDOMLY]"

	else:
		with numpy.load(oraclenamein) as oraclein:
			weights = [oraclein['weightsin'], oraclein['weightsout']]
			biases = [oraclein['biasesin'], oraclein['biasesout']]
		if configuration.verbose:
			print "[DEBUG: ORACLE INITIALIZED FROM FILE: %s]" %oraclenamein

	return {'weights': weights, 'biases': biases}

def writeoracle(oraclenameout):

	if oraclenameout is None:
		print "weightsin", oracle['weights'][0], "weightsout", oracle['weights'][1], "biasesin", oracle['biases'][0], "biasesout", oracle['biases'][1]
	else:
		numpy.savez(oraclenameout, weightsin = oracle['weights'][0], weightsout = oracle['weights'][1], biasesin = oracle['biases'][0], biasesout = oracle['biases'][1])
		if configuration.verbose:
			print "[DEBUG: ORACLE WRITTEN TO FILE: %s]" %oraclenameout

if __name__ == '__main__':

	oraclenameout, oraclenamein, translatorfile, vecnamein, namein, dirnamein, labels = arguments(sys.argv[1:])
	vectorspace, vectorsize = translate.vectorspace(vecnamein)
	oracle = readoracle(oraclenamein, len(labels))
	if vectorsize != configuration.embeddingsize:
		configuration.error("[ERROR: EMBEDDING SIZE MISMATCH]")

	if translatorfile is not None:
		translator = translate.readtranslator(translatorfile)
		vectorspace = translate.translatevectorspace(vectorspace, translator)

	trainingcount = 0

	if dirnamein is None:
		if namein is None:
			configuration.error(usage)

		if configuration.filetype == "Stanford":
			sentences, words, count = dependencies.readStanford(namein)
		elif configuration.filetype == "CONLL":
			sentences, words, count = dependencies.readCONLL(namein)
		else:
			configuration.error("[EXIT: UNSUPPORTED DEPENDENCY FILETYPE]")
		vectors = translate.wordstovectors(words, vectorspace)
		for i in range(configuration.iterations):
			oracle, count = trainshiftreducer(oracle, labels, sentences, vectors)
			trainingcount += count

	else:
		for root, dirs, files in os.walk(dirnamein, topdown = False):
			for name in files:
				if configuration.filetype == "Stanford":
					sentences, words, count = dependencies.readStanford(os.path.join(root, name))
				elif configuration.filetype == "CONLL":
					sentences, words, count = dependencies.readCONLL(os.path.join(root, name))
				else:
					configuration.error("[EXIT: UNSUPPORTED DEPENDENCY FILETYPE]")
				vectors = translate.wordstovectors(words, vectorspace)
				for i in range(configuration.iterations):
					oracle, count = trainshiftreducer(oracle, labels, sentences, vectors)
					trainingcount += count

	if configuration.train:
		if configuration.verbose:
			print "[DEBUG: ORACLE TRAINED ON %d PARSED INSTANCES]" %trainingcount
		writeoracle(oraclenameout)
