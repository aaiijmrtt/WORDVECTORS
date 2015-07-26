import re, sys

verbose = False
find = False
train = True

embeddingsize = 200
minlength = 5
maxlength = 100
tokensize = 100
floatsize = 4
contextnumber = 4
iterations = 1

filetype = "Stanford"

permissible = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz \t\n")

tags = re.compile("<TEXT>([^<]*)</TEXT>")
glob = re.compile("([^\Z]*)")
tran = re.compile("^([^\t]*)\t([^\n]*)\n")
head = re.compile("^([^ ]*) ([^\n]*)(\n)")
entry = re.compile("^([^ ]*)( )")
parse = re.compile("(.*?)\((.*?), (.*?)\)")

def error(description):

	print description
	sys.exit()
