#!/usr/bin/python
import sys, getopt, re, numpy, string, os
from translate import *
from neuralnet import *
from shiftreducer import *

usage = "\n\nNAME\n\n\n\toracles - train parser oracles\n\n\nSYNOPSIS\n\n\n\t./oracles.py [OPTIONS]... <FILE:VEC> <FILE|DIR:PAR> <ARRAY>...\n\n\nDESCRIPTION\n\n\n\tTrain shift reduce dependency parser oracles. Use neural network language model. Learn phrase vectors automatically. Print to stdout unless write file specified.\n\n\tMandatory arguments to long options are mandatory for short options too.\n\n\nOPTIONS\n\n\t-h, --help\n\t\tdisplay this help and exit\n\n\t-v, --verbose\n\t\tdisplay information for debugging\n\n\t-s, --sentence=INT\n\t\taccept maximum sentence length (default is 100)\n\n\t-e, --embedding=INT\n\t\taccept size of vector space embedding (default is 200)\n\n\t-w, --write=FILE\n\t\taccept file for writing\n\n\nARGUMENTS\n\n\tFILE:VEC\n\t\taccept file containing word vectors\n\n\tFILE:PAR\n\t\taccept file containing Standford typed dependencies\n\n\tDIR:PAR\n\t\taccept directory containing Standford typed dependencies\n\n\tARRAY\n\t\taccept dependency labels for training\n\n\nAUTHOR\n\n\n\tWritten by Amitrajit Sarkar.\n\n\nREPORTING BUGS\n\n\n\tReport bugs to <aaiijmrtt@gmail.com>.\n\n\nCOPYRIGHT\n\n\nThe MIT License (MIT)\n\nCopyright (c) 2015 Amitrajit Sarkar\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"

verbose = False
embeddingsize = 200
maxlength = 100

PARSER_PATTERN = re.compile("^([a-z:]*)\(([a-zA-Z]*)-([0-9]*), ([a-zA-Z]*)-([0-9]*)\)\n$")
SHIFT = -1
REDUCELEFT = 0
REDUCERIGHT = 1

NONLINEARITY = lambda x: 1.0 / (1.0 + math.exp(-x))
NONLINEARITY_MAX = 1.0
NONLINEARITY_MIN = 0.0

def error(description):

	print description
	sys.exit()

def arguments(string):

	global verbose, embeddingsize, maxlength
	labels = None
	vecnamein = None
	namein = None
	dirnamein = None
	write = False

	try:
		opts, args = getopt.getopt(string, "hvs:e:w", ["help=", "verbose=", "sentence=", "embedding="])

		for opt, arg in opts:
			if opt in ("-h", "--help"):
				error(usage)
			elif opt in ("-v", "--verbose"):
				verbose = True
			elif opt in ("-s", "--sentence"):
				maxlength = int(arg)
			elif opt in ("-e", "--embedding"):
				embeddingsize = int(arg)
			elif opt in ("-w", "--write"):
				write = True

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

	if verbose:
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT sentence: %d ]" %maxlength
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT embedding: %s ]" %embeddingsize
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT write: %s ]" %write
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT vectorfile: %s ]" %vecnamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsefile: %s ]" %namein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsedirectory: %s ]" %dirnamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT labels: %s ]" %labels

	return write, vecnamein, namein, dirnamein, labels

def initializeoracles(embeddingsize, labels):

	contextnumber = 2
	weights = [numpy.random.rand(embeddingsize, contextnumber * embeddingsize), numpy.random.rand(1, embeddingsize)]
	biases = [numpy.random.rand(embeddingsize, 1), numpy.random.rand(1, 1)]
	oracleshift = {'weights': weights, 'biases': biases, 'label': 'shift'}
	oraclesreduceleft = list()
	oraclesreduceright = list()
	for label in labels:
		weights = [numpy.random.rand(embeddingsize, contextnumber * embeddingsize), numpy.random.rand(1, embeddingsize)]
		biases = [numpy.random.rand(embeddingsize, 1), numpy.random.rand(1, 1)]
		oraclesreduceleft.append({'weights': weights, 'biases': biases, 'label': label})
		weights = [numpy.random.rand(embeddingsize, contextnumber * embeddingsize), numpy.random.rand(1, embeddingsize)]
		biases = [numpy.random.rand(embeddingsize, 1), numpy.random.rand(1, 1)]
		oraclesreduceright.append({'weights': weights, 'biases': biases, 'label': label})
	weights = [numpy.random.rand(embeddingsize, contextnumber * embeddingsize), numpy.random.rand(1, embeddingsize)]
	biases = [numpy.random.rand(embeddingsize, 1), numpy.random.rand(1, 1)]
	oraclesreduceleft.append({'weights': weights, 'biases': biases, 'label': 'reduce'})
	weights = [numpy.random.rand(embeddingsize, contextnumber * embeddingsize), numpy.random.rand(1, embeddingsize)]
	biases = [numpy.random.rand(embeddingsize, 1), numpy.random.rand(1, 1)]
	oraclesreduceright.append({'weights': weights, 'biases': biases, 'label': 'reduce'})
	return oracleshift, oraclesreduceleft, oraclesreduceright

def trainshiftreducer(oracleshift, oraclesreduceleft, oraclesreduceright, dependencyfile, vectorspace):
	trainingcount = 0
	while dependencyfile.tell() != os.fstat(dependencyfile.fileno()).st_size:
		endoftreeflag = False
		validtreeflag = True
		inputs = [None for i in range(maxlength)]
		outputs = list()
		wordcount = 0
		while not endoftreeflag:
			readline = dependencyfile.readline()
			match = PARSER_PATTERN.match(readline)
			if match:
				validlabelflag = False
				if match.group(2).lower() in vectorspace and match.group(2) != 'ROOT':
					inputs[int(match.group(3))] = vectorspace[match.group(2).lower()]
				elif match.group(2).lower() not in vectorspace and match.group(2) != 'ROOT':
					validtreeflag = False
				if match.group(4).lower() in vectorspace and match.group(4) != 'ROOT':
					inputs[int(match.group(5))] = vectorspace[match.group(4).lower()]
				elif match.group(4).lower() not in vectorspace and match.group(4) != 'ROOT':
					validtreeflag = False
				for oracle in oraclesreduceleft:
					if oracle['label'] == match.group(1):
						validlabelflag = True
				if validlabelflag:
					outputs.append((int(match.group(3)), int(match.group(5)), match.group(1)))
				else:
					outputs.append((int(match.group(3)), int(match.group(5)), 'reduce'))
				wordcount = max(wordcount, int(match.group(3)), int(match.group(5)))
			else:
				endoftreeflag = True
		if validtreeflag:
			inputs[0] = vectorspace['</s>']
			inputs = [numpy.reshape(inp, (embeddingsize, 1)) for inp in inputs if inp is not None]
			if len(inputs) != wordcount + 1:
				continue
			oracleshift, oraclesreduceleft, oraclesreduceright = trainoracles(inputs, outputs, oracleshift, oraclesreduceleft, oraclesreduceright)
			trainingcount += 1
	if verbose:
		print "[DEBUG: ORACLES TRAINED ON %d PARSED SENTENCES IN FILE %s]" %(trainingcount, dependencyfile.name)
	return oracleshift, oraclesreduceleft, oraclesreduceright, trainingcount

if __name__ == '__main__':

	write, vecnamein, namein, dirnamein, labels = arguments(sys.argv[1:])
	oracleshift, oraclesreduceleft, oraclesreduceright = initializeoracles(embeddingsize, labels)
	vectorspace, vectorsize = vectorspace(vecnamein)
	if vectorsize != embeddingsize:
		error("[ERROR: EMBEDDING SIZE MISMATCH]")
	trainingcount = 0

	if dirnamein is None:
		if namein is None:
			error(usage)
		with open(namein, "r") as filein:
			oracleshift, oraclesreduceleft, oraclesreduceright, count = trainshiftreducer(oracleshift, oraclesreduceleft, oraclesreduceright, filein, vectorspace)
			trainingcount += count
	else:
		for root, dirs, files in os.walk(dirnamein, topdown = False):
			for name in files:
				with open(os.path.join(root, name), "r") as filein:
					oracleshift, oraclesreduceleft, oraclesreduceright, count = trainshiftreducer(oracleshift, oraclesreduceleft, oraclesreduceright, filein, vectorspace)
					trainingcount += count
	if verbose:
		print "[DEBUG: ORACLES TRAINED ON %d PARSED SENTENCES]" %trainingcount

	if write:
		numpy.savez("SHIFT", weightsin = oracleshift['weights'][0], weightsout = oracleshift['weights'][1], biasesin = oracleshift['biases'][0], biasesout = oracleshift['biases'][1])
		for oracle in oraclesreduceleft:
			numpy.savez("LEFT" + oracle['label'], weightsin = oracle['weights'][0], weightsout = oracle['weights'][1], biasesin = oracle['biases'][0], biasesout = oracle['biases'][1])
		for oracle in oraclesreduceright:
			numpy.savez("RIGHT" + oracle['label'], weightsin = oracle['weights'][0], weightsout = oracle['weights'][1], biasesin = oracle['biases'][0], biasesout = oracle['biases'][1])
		if verbose:
			print "[DEBUG: ORACLES WRITTEN TO FILES]"
	else:
		print "SHIFT", "weightsin", oracleshift['weights'][0], "weightsout", oracleshift['weights'][1], "biasesin", oracleshift['biases'][0], "biasesout", oracleshift['biases'][1]
		for oracle in oraclesreduceleft:
			print "LEFT" + oracle['label'], "weightsin", oracle['weights'][0], "weightsout", oracle['weights'][1], "biasesin", oracle['biases'][0], "biasesout", oracle['biases'][1]
		for oracle in oraclesreduceright:
			print "RIGHT" + oracle['label'], "weightsin", oracle['weights'][0], "weightsout", oracle['weights'][1], "biasesin", oracle['biases'][0], "biasesout", oracle['biases'][1]
