#!/usr/bin/python
import sys, getopt, re, numpy

usage = "\n\nNAME\n\n\n\ttranslate - process monolingual word vectors and bilingual dictionary\n\n\nSYNOPSIS\n\n\n\t./translate.py [OPTIONS]... <FILE:SRCVEC> <FILE:TGTVEC> <FILE:BIDIC>...\n\n\nDESCRIPTION\n\n\n\tProcess monolingual word vectors and bilingual dictionary. Generate translation matrix. Print to stdout unless write file specified.\n\n\tMandatory arguments to long options are mandatory for short options too.\n\n\nOPTIONS\n\n\t-h, --help\n\t\tdisplay this help and exit\n\n\t-v, --verbose\n\t\tdisplay information for debugging\n\n\t-w, --write=FILE\n\t\taccept file for writing\n\n\t-t, --translator=FILE\n\t\taccept file containing translator matrix\n\n\nARGUMENTS\n\n\tFILE:SRCVEC\n\t\taccept file containing source language word vectors\n\n\tFILE:TGTVEC\n\t\taccept file containing target language word vectors\n\n\tFILE:BIDIC\n\t\taccept file containing dictionary from source to target language\n\n\nAUTHOR\n\n\n\tWritten by Amitrajit Sarkar.\n\n\nREPORTING BUGS\n\n\n\tReport bugs to <aaiijmrtt@gmail.com>.\n\n\nCOPYRIGHT\n\n\nThe MIT License (MIT)\n\nCopyright (c) 2015 Amitrajit Sarkar\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"

head = re.compile("^([^ ]*) ([^\n]*)(\n)")
entry = re.compile("^([^ ]*)( )")
tran = re.compile("^([^\t]*)\t([^\n]*)\n")
floatsize = 4
verbose = False

def error(description):

	print description
	sys.exit()

def arguments(string):

	global verbose
	nameout = None
	sourcevectors = None
	targetvectors = None
	transferwords = None
	translator = None

	try:
		opts, args = getopt.getopt(string, "hvw:t:", ["help", "verbose", "write=", "translator="])
	except getopt.GetoptError:
		error(usage)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			error(usage)
		elif opt in ("-v", "--verbose"):
			verbose = True
		elif opt in ("-w", "--write"):
			nameout = arg
		elif opt in ("-t", "--translator"):
			translator = arg

	if len(args) != 3:
		error(usage)
	else:
		sourcevectors = args[0]
		targetvectors = args[1]
		transferwords = args[2]

	if verbose:
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT write: %s]" %nameout
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT translator: %s]" %translator
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT sourcevectors: %s]" %sourcevectors
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT targetvectors: %s]" %targetvectors
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT dictionary: %s]" %transferwords

	return nameout, translator, sourcevectors, targetvectors, transferwords

def dictionary(filename):

	trans = dict()
	with open(filename, "r") as transfile:
		for line in transfile.readlines():
			match = tran.match(line)
			if match is None:
				error("[EXIT: DICTIONARY FORMAT ERROR]")
			else:
				trans[match.group(1).lower()] = match.group(2).lower()

	if verbose:
		print "[DEBUG: TRANSLATION DICTIONARY INITIALIZED BY: %s]" %filename

	return trans

def vectorspace(filename):

	space = dict()
	with open(filename, "rb") as vecbin:
		bytes = vecbin.read()
		match = head.match(bytes)
		if match is None:
			error("[EXIT: BINARY FORMAT ERROR]")
		vocabularysize = int(match.group(1))
		embeddingsize = int(match.group(2))
		sindex = match.start(3) + 1

		for i in range(vocabularysize):
			match = entry.match(bytes[sindex: ])
			if match is None:
				error("[EXIT: BINARY FORMAT ERROR]")
			findex = match.start(2) + 1
			space[match.group(1).lower()] = numpy.fromstring(bytes[sindex + findex: ], dtype = numpy.float32, count = embeddingsize)
			sindex += findex + embeddingsize * floatsize + 1

	if verbose:
		print "[DEBUG: VECTOR SPACE INITIALIZED BY: %s]" %filename

	return space, embeddingsize

def traintranslator(source, svectorsize, destination, tvectorsize, dictionary, translator = None, alpha = 0.05):

	translator = numpy.random.rand(tvectorsize, svectorsize) if translator is None else translator
	for word in dictionary:
		if word in source and dictionary[word] in destination:
			invector = source[word]
			outvector = destination[dictionary[word]]
			activation = numpy.dot(translator, invector)
			difference = numpy.subtract(activation, outvector)
			delta = numpy.dot(difference, invector.transpose())
			delta = numpy.multiply(alpha, delta)
			translator = numpy.subtract(translator, delta)

	if verbose:
		print "[DEBUG: TRANSLATION MATRIX GENERATED]"

	return translator

if __name__ == '__main__':

	nameout, translatorfile, sourcevectors, targetvectors, transferwords = arguments(sys.argv[1: ])
	source, svectorsize = vectorspace(sourcevectors)
	target, tvectorsize = vectorspace(targetvectors)
	dictionary = dictionary(transferwords)
	translator = None if translatorfile is None else numpy.load(translatorfile, "r+")
	translator = traintranslator(source, svectorsize, target, tvectorsize, dictionary, translator)

	if nameout is None:
		print translator
	else:
		numpy.save(nameout, translator)
		if verbose:
			print "[DEBUG: TRANSLATION MATRIX SAVED TO: %s.npy]" %nameout
