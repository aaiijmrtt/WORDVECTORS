import re

verbose = False
find = False
phraser = False

embeddingsize = 200
maxlength = 100
tokensize = 100
floatsize = 4
contextnumber = 2
iterations = 1

tags = re.compile("<TEXT>([^<]*)</TEXT>")
glob = re.compile("([^\Z]*)")
tran = re.compile("^([^\t]*)\t([^\n]*)\n")
head = re.compile("^([^ ]*) ([^\n]*)(\n)")
entry = re.compile("^([^ ]*)( )")
parse = re.compile("(.*?)\((.*?), (.*?)\)\n")
