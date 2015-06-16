#!/usr/bin/python
import sys, os, getopt, codecs, re

usage = "\n\nNAME\n\n\n\tcorpus - process corpus\n\n\nSYNOPSIS\n\n\n\t./corpus.py [OPTIONS]... <FILE|DIR>...\n\n\nDESCRIPTION\n\n\n\tProcess corpus. Transliterate if specified. If from the FIRE corpus, strip <TAGS> and extract text </TAGS>. Print to stdout unless write file specified.\n\n\tMandatory arguments to long options are mandatory for short options too.\n\n\nOPTIONS\n\n\t-h, --help\n\t\tdisplay this help and exit\n\n\t-v, --verbose\n\t\tdisplay information for debugging\n\n\t-f, --fire\n\t\tassume FILE|DIR in FIRE corpus document format\n\n\t-w, --write=FILE\n\t\taccept file for writing\n\n\t-t, --transliterate=FILE\n\t\taccept file for transliteration\n\n\nARGUMENTS\n\n\tFILE\n\t\taccept file for preprocessing\n\n\tDIR\n\t\taccept directory for preprocessing\n\n\nAUTHOR\n\n\n\tWritten by Amitrajit Sarkar.\n\n\nREPORTING BUGS\n\n\n\tReport bugs to <aaiijmrtt@gmail.com>.\n\n\nCOPYRIGHT\n\n\nThe MIT License (MIT)\n\nCopyright (c) 2015 Amitrajit Sarkar\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"

tags = re.compile("<TEXT>([^<]*)</TEXT>")
glob = re.compile("([^\Z]*)")
tran = re.compile("^([^\t]*)\t([^\n]*)\n")
verbose = False
fire = False

def error(description):

	print description
	sys.exit()

def arguments(string):

	global verbose, fire
	namein = None
	nameout = None
	dirnamein = None
	transname = None

	try:
		opts, args = getopt.getopt(string, "hvfw:t:", ["help", "verbose", "fire", "write=", "transliterate="])
	except getopt.GetoptError:
		error(usage)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			error(usage)
		elif opt in ("-v", "--verbose"):
			verbose = True
		elif opt in ("-f", "--fire"):
			fire = True
		elif opt in ("-w", "--write"):
			nameout = arg
		elif opt in ("-t", "--transliterate"):
			transname = arg

	if len(args) != 1:
		error(usage)

	if os.path.isdir(args[0]):
		dirnamein = args[0]
	else:
		namein = args[0]

	if verbose:
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT read: %s ]" %namein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT write: %s ]" %nameout
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT directory: %s ]" %dirnamein
		print "[DEBUG: PARSED COMMANDLINE ARGUMENT transliterate: %s ]" %transname

	return namein, nameout, dirnamein, transname

def translit(filetrans):

	trans = dict()
	with codecs.open(filetrans, "r", "utf-8") as transfile:
		for line in transfile.readlines():
			match = tran.match(line)
			if match is None:
				error("[EXIT: DICTIONARY FORMAT ERROR]")
			else:
				trans[match.group(1)] = match.group(2)

	if verbose:
		print "[DEBUG: TRANSLITERATED BY file: %s ]" %filetrans

	return trans

def readwrite(filein, fileout, trans = None):

	try:
		text = filein.read()
	except Exception:
		print "[ERROR: DOCUMENT ENCODING ERROR: %s]" %filein.name
		return

	match = tags.search(text) if fire else glob.match(text)
	if match is None:
		print "[ERROR: DOCUMENT FORMAT ERROR: %s]" %filein.name
		return

	if trans is None:
		fileout.write(match.group(1))
	else:
		for tokenin in match.group(1):
			tokenout = trans[tokenin] if tokenin in trans else tokenin
			fileout.write(tokenout)

	if verbose:
		print "[DEBUG: PREPROCESSED file: %s ]" %filein.name

if __name__ == '__main__':

	namein, nameout, dirnamein, transname = arguments(sys.argv[1:])
	fileout = sys.stdout if nameout is None else codecs.open(nameout, "w", "utf-8")
	trans = translit(transname) if transname is not None else None

	if dirnamein is None:
		if namein is None:
			error(usage)
		with codecs.open(namein, "r", "utf-8") as filein:
			readwrite(filein, fileout, trans)
	else:
		for root, dirs, files in os.walk(dirnamein, topdown = False):
			for name in files:
				with codecs.open(os.path.join(root, name), "r", "utf-8") as filein:
					readwrite(filein, fileout, trans)

	if fileout is not sys.stdout:
		fileout.close()
