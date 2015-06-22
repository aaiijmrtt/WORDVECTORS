# WORDVECTORS
exploring deep learning: natural language processing

#Using APIWrapper java for extracting typedDependencies from txt file
#1. Compiling
javac -cp stanford-parser.jar APIWrapper.java
#2. Executing
java -cp .:stanford-parser.jar:stanford-parser-3.5.2-models.jar APIWrapper <text_file> >> out.txt
