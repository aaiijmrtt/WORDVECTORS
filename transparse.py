#!/usr/bin/python
import sys, getopt, re, numpy, os
import configuration, translate, oracles, neuralnet, shiftreducer

usage = "\n\nNAME\n\n\n\ttransparse - process word vectors for translating and dependency parsing\n\n\nSYNOPSIS\n\n\n\t./transparse.py [OPTIONS]... <FILE:VEC> <FILE:ORACLE> <FILE|DIR:PAR>\n\n\nDESCRIPTION\n\n\n\tProcess word vectors for dependency parsing. Translate word vectors if specified. Print to stdout unless write file specified.\n\n\tMandatory arguments to long options are mandatory for short options too.\n\n\nOPTIONS\n\n\n\t-h, --help\n\t\tdisplay this help and exit\n\n\t-s, --size=INT[100]\n\t\taccept maximum size of tokens in vectorspace\n\n\t-t, --translator=FILE\n\t\taccept file containing translator\n\n\t-v, --verbose\n\t\tdisplay information for debugging\n\n\t-w, --write=FILE\n\t\taccept file for writing\n\n\nARGUMENTS\n\n\n\tFILE:VEC\n\t\taccept file containing word vectors\n\n\tFILE:ORACLE\n\t\taccept file containing shift reduce parser oracle\n\n\tFILE:PAR\n\t\taccept file containing sentences to be parsed\n\n\tDIR:PAR\n\t\taccept directory containing sentences to be parsed\n\n\nAUTHOR\n\n\n\tWritten by Amitrajit Sarkar.\n\n\nREPORTING BUGS\n\n\n\tReport bugs to <aaiijmrtt@gmail.com>.\n\n\nCOPYRIGHT\n\n\nThe MIT License (MIT)\n\nCopyright (c) 2015 Amitrajit Sarkar\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"

def arguments(string):

	nameout = None
	translator = None
	sourcevectors = None
	oraclenamein = None
	parsefile = None
	parsedirectory = None

	try:
		opts, args = getopt.getopt(string, "hvw:t:s:", ["help", "verbose", "write=", "translator=", "size="])
		for opt, arg in opts:
			if opt in ("-h", "--help"):
				configuration.error(usage)
			elif opt in ("-v", "--verbose"):
				configuration.verbose = True
			elif opt in ("-w", "--write"):
				nameout = arg
			elif opt in ("-t", "--translator"):
				translator = arg
			elif opt in ("-s", "--size"):
				configuration.tokensize = int(arg)

	except Exception:
		configuration.error(usage)

	if len(args) != 3:
		configuration.error(usage)

	sourcevectors = args[0]
	oraclenamein = args[1]
	if os.path.isdir(args[2]):
		parsedirectory = args[2]
	else:
		parsefile = args[2]

	if configuration.verbose:
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT write: %s]" %nameout
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT translator: %s]" %translator
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT size: %d]" %configuration.tokensize
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT vectors: %s]" %sourcevectors
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT oracle: %s]" %oraclenamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsefile: %s]" %parsefile
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT parsedirectory: %s]" %parsedirectory

	return nameout, translator, sourcevectors, oraclenamein, parsefile, parsedirectory

def parse(filein, fileout, oracle, translator = None):

	parsedcount = 0
	droppedcount = 0

	for line in filein.readlines():
		inputs = list()
		check = True

		for word in line.split():
			if word.lower() in source:
				vector = source[word.lower()] if translator is None else translate.translatevector(source[word.lower()], translator)
				inputs.append(vector)
			else:
				check = False
				if configuration.verbose:
					print "[DEBUG: WORD %s NOT IN VECTORSPACE]" %word
				break

		if check:
			fileout.write(str(shiftreducer.shiftreduce(inputs, oracle)) + "\n")
			parsedcount += 1
		else:
			droppedcount += 1

	if configuration.verbose:
		print "[DEBUG: PARSED %d SENTENCES IN FILE %s]" %(parsedcount, filein.name)
		print "[DEBUG: DROPPED %d SENTENCES IN FILE %s]" %(droppedcount, filein.name)

if __name__ == '__main__':

	nameout, translation, sourcevectors, oraclenamein, parsefile, parsedirectory = arguments(sys.argv[1: ])
	source, svectorsize = translate.vectorspace(sourcevectors)
	oracle = oracles.readoracle(oraclenamein)
	translator = None if translation is None else translate.readtranslator(translation)
	fileout = sys.stdout if nameout is None else open(nameout, "w")

	if parsedirectory is None:
		if parsefile is None:
			configuration.error(usage)
		with open(parsefile, "r") as filein:
			parse(filein, fileout, oracle, translator)
	else:
		for root, dirs, files in os.walk(parsedirectory, topdown = False):
			for name in files:
				with open(os.path.join(root, name), "r") as filein:
					parse(filein, fileout, oracle, translator)

	if fileout is not sys.stdout:
		fileout.close()
