#!/usr/bin/python
import sys, getopt, random, math, numpy, scipy.spatial.distance
import configuration, neuralnet

usage = "\n\nNAME\n\n\n\ttranslate - process monolingual word vectors and bilingual dictionary\n\n\nSYNOPSIS\n\n\n\t./translate.py [OPTIONS]... <FILE:SRCVEC> <FILE:TGTVEC> <FILE:BIDIC>\n\n\nDESCRIPTION\n\n\n\tProcess monolingual word vectors and bilingual dictionary. Generate translation matrix. Print to stdout unless write file specified.\n\n\tMandatory arguments to long options are mandatory for short options too.\n\n\nOPTIONS\n\n\n\t-h, --help\n\t\tdisplay this help and exit\n\n\t-i, --iterations=INT[1]\n\t\taccept number of iterations to train for\n\n\t-s, --size=INT[100]\n\t\taccept maximum size of tokens in vectorspace\n\n\t-t, --translator=FILE\n\t\taccept file containing translator\n\n\t-v, --verbose\n\t\tdisplay information for debugging\n\n\t-w, --write=FILE\n\t\taccept file for writing\n\n\nARGUMENTS\n\n\n\tFILE:SRCVEC\n\t\taccept file containing source language word vectors\n\n\tFILE:TGTVEC\n\t\taccept file containing target language word vectors\n\n\tFILE:BIDIC\n\t\taccept file containing dictionary from source to target language\n\n\nAUTHOR\n\n\n\tWritten by Amitrajit Sarkar.\n\n\nREPORTING BUGS\n\n\n\tReport bugs to <aaiijmrtt@gmail.com>.\n\n\nCOPYRIGHT\n\n\nThe MIT License (MIT)\n\nCopyright (c) 2015 Amitrajit Sarkar\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"

VECTORIZEDDISTANCE = scipy.spatial.distance.cosine

def error(description):

	print description
	sys.exit()

def arguments(string):

	nameout = None
	sourcevectors = None
	targetvectors = None
	transferwords = None
	translator = None

	try:
		opts, args = getopt.getopt(string, "hvw:t:s:i:", ["help", "verbose", "write=", "translator=", "size=", "iterations="])
		for opt, arg in opts:
			if opt in ("-h", "--help"):
				error(usage)
			elif opt in ("-v", "--verbose"):
				configuration.verbose = True
			elif opt in ("-w", "--write"):
				nameout = arg
			elif opt in ("-t", "--translator"):
				translator = arg
			elif opt in ("-s", "--size"):
				configuration.tokensize = int(arg)
			elif opt in ("-i", "--iterations"):
				configuration.iterations = int(arg)

	except Exception:
		error(usage)

	if len(args) != 3:
		error(usage)

	sourcevectors = args[0]
	targetvectors = args[1]
	transferwords = args[2]

	if configuration.verbose:
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT write: %s]" %nameout
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT translator: %s]" %translator
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT size: %d]" %configuration.tokensize
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT iterations: %d]" %configuration.iterations
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT sourcevectors: %s]" %sourcevectors
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT targetvectors: %s]" %targetvectors
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT dictionary: %s]" %transferwords

	return nameout, translator, sourcevectors, targetvectors, transferwords

def dictionary(filename):

	count = 0
	trans = dict()

	with open(filename, "r") as transfile:
		for line in transfile.readlines():
			match = configuration.tran.match(line)
			if match is None:
				error("[EXIT: DICTIONARY FORMAT ERROR]")
			else:
				trans[match.group(1).lower()] = match.group(2).lower()
			count += 1

	if configuration.verbose:
		print "[DEBUG: TRANSLATION DICTIONARY INITIALIZED BY: %s]" %filename
		print "[DEBUG: TRANSLATION DICTIONARY SIZE: %d]" %count

	return trans

def vectorspace(filename):

	space = dict()

	with open(filename, "rb") as vecbin:
		bytes = vecbin.read()
		match = configuration.head.match(bytes)
		if match is None:
			error("[EXIT: BINARY FORMAT ERROR]")
		vocabularysize = int(match.group(1))
		embeddingsize = int(match.group(2))
		sindex = match.start(3) + 1

		for i in range(vocabularysize):
			match = configuration.entry.match(bytes[sindex: sindex + configuration.tokensize])
			if match is None:
				error("[EXIT: BINARY FORMAT ERROR]")
			findex = match.start(2) + 1
			space[match.group(1).lower()] = numpy.reshape(numpy.fromstring(bytes[sindex + findex: sindex + findex + embeddingsize * configuration.floatsize], dtype = numpy.float32, count = embeddingsize), (embeddingsize, 1))
			sindex += findex + embeddingsize * configuration.floatsize + 1

	if configuration.verbose:
		print "[DEBUG: VECTOR SPACE INITIALIZED FROM FILE: %s]" %filename
		print "[DEBUG: VECTOR SPACE COUNT: %d]" %vocabularysize

	return space, embeddingsize

def scalevectorspace(vectorspace, vectorsize):

	maximum = float("-inf")
	minimum = float("inf")
	for word in vectorspace:
		maximum = max(maximum, numpy.max(vectorspace[word]))
		minimum = min(minimum, numpy.min(vectorspace[word]))
	shift = - minimum
	scale = maximum - minimum

	vectorizedscaler = numpy.vectorize(lambda x: (x + shift) / scale)
	for word in vectorspace:
		vectorspace[word] = vectorizedscaler(vectorspace[word])

	if configuration.verbose:
		print "[DEBUG: SHIFTED VECTORSPACE BY %f]" %shift
		print "[DEBUG: SCALED VECTORSPACE BY %f]" %scale

	return vectorspace, shift, scale

def traintranslator(source, svectorsize, destination, tvectorsize, dictionary, translator, alpha = 0.05):

	inputs = list()
	outputs = list()

	for word in dictionary:
		if word in source and dictionary[word] in destination:
			inputs.append(source[word])
			outputs.append(destination[dictionary[word]])

	altogether = list(zip(inputs, outputs))
	random.shuffle(altogether)
	inputs, outputs = zip(*altogether)
	traininputs = inputs[0: int(0.80 * len(inputs))]
	trainoutputs = outputs[0: int(0.80 * len(outputs))]
	testinputs = inputs[int(0.80 * len(inputs)): ]
	testoutputs = outputs[int(0.80 * len(outputs)): ]

	for i in range(configuration.iterations):
		translator['weights'], translator['biases'] = neuralnet.train(traininputs, trainoutputs, translator['weights'], translator['biases'])
		error = neuralnet.test(testinputs, testoutputs, translator['weights'], translator['biases']) * 2.0 / tvectorsize

		if configuration.verbose:
			print "[DEBUG: TRANSLATOR TRAINED ON %d WORDPAIRS]" %len(traininputs)
			print "[DEBUG: TRANSLATOR TESTED ON %d WORDPAIRS]" %len(testinputs)
			print "[DEBUG: TRANSLATOR ERROR %f]" %error

	return translator, len(inputs), error

def translatevector(sourcevector, translator, vectorizednonlinearity = neuralnet.VECTORIZEDNONLINEARITY):

	unscaledvector = vectorizednonlinearity(numpy.add(numpy.dot(translator['weights'][0], sourcevector), translator['biases'][0]))
	vectorizedscaler = numpy.vectorize(lambda x: x * translator['scale'] - translator['shift'])
	return vectorizedscaler(unscaledvector)

def translateword(sourceword, sourcespace, translator, targetspace, vectorizednonlinearity = neuralnet.VECTORIZEDNONLINEARITY, vectorizeddistance = VECTORIZEDDISTANCE, neighbourcount = 5):

	return findneighbours(translatevector(sourcespace[sourceword], translator, vectorizednonlinearity), targetspace, vectorizeddistance, neighbourcount)

def findneighbours(vector, vectorspace, vectorizeddistance = VECTORIZEDDISTANCE, neighbourcount = 5):

	count = 0
	neighbours = list()
	distances = list()
	distancethreshold = float("inf")

	for word in vectorspace:
		distance = math.fabs(numpy.sum(vectorizeddistance(vector, vectorspace[word])))
		if distance < distancethreshold:
			if len(neighbours) == 0:
				distances.append(distance)
				neighbours.append(word)

			else:
				inserted = False
				for i in reversed(range(len(distances))):
					if distances[i] < distance:
						distances.insert(i + 1, distance)
						neighbours.insert(i + 1, word)
						inserted = True
						break

				if not inserted:
					distances.insert(0, distance)
					neighbours.insert(0, word)

			if len(neighbours) > neighbourcount:
				distances = distances[0: neighbourcount]
				neighbours = neighbours[0: neighbourcount]
				distancethreshold = distances[-1]

	return neighbours, distances

def readtranslator(translatorfile, svectorsize = None, tvectorsize = None, shift = None, scale = None):

	if translatorfile is None:
		weights = [numpy.random.rand(tvectorsize, svectorsize)]
		biases = [numpy.random.rand(tvectorsize, 1)]
		if configuration.verbose:
			print "[DEBUG: TRANSLATOR INITIALIZED RANDOMLY]"

	else:
		with numpy.load(translatorfile, "r") as translationfile:
			weights = [translationfile['weights']]
			biases = [translationfile['biases']]
			shift = translationfile['shift']
			scale = translationfile['scale']
			if configuration.verbose:
				print "[DEBUG: TRANSLATOR INITIALIZED FROM FILE: %s]" %translatorfile

	return {'weights': weights, 'biases': biases, 'shift': shift, 'scale': scale}

def writetranslator(translator, translatorfile):

	if translatorfile is None:
		print translator
	else:
		numpy.savez(translatorfile, weights = translator['weights'][0], biases = translator['biases'][0], shift = translator['shift'], scale = translator['scale'])
		if configuration.verbose:
			print "[DEBUG: TRANSLATOR SAVED TO FILE: %s.npy]" %translatorfile

if __name__ == '__main__':

	nameout, translatorfile, sourcevectors, targetvectors, transferwords = arguments(sys.argv[1: ])
	source, svectorsize = vectorspace(sourcevectors)
	target, tvectorsize = vectorspace(targetvectors)
	target, shift, scale = scalevectorspace(target, tvectorsize)
	dictionary = dictionary(transferwords)
	translator = readtranslator(translatorfile, svectorsize, tvectorsize, shift, scale)
	translator, count, error = traintranslator(source, svectorsize, target, tvectorsize, dictionary, translator)
	writetranslator(translator, nameout)
