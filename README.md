#WORDVECTORS
exploring deep learning: natural language processing

###Java Sources:

1. **APIWrapper.java**: for extracting typedDependencies from txt file

**Instructions**:

* **Compiling**: $javac -cp stanford-parser.jar APIWrapper.java
* **Executing**: $java -cp .:stanford-parser.jar:stanford-parser-3.5.2-models.jar APIWrapper in.txt >> out.txt

**Notes**:

* Requires Java 8.
* Requires Standford Parser jars.

###Python Scripts:

1. **corpus.py**: a script to preprocess the corpus
2. **translate.py**: a script to learn the translation matrix between vector spaces
3. **oracles.py**: a script to train the shift reduce dependency parser oracles
4. **transparse.py**: a script to parse dependencies, after vector translation if specified

**Instructions**:

* **Setting Executable Bit**: eg. $chmod +x corpus.py
* **Browsing Usage**: eg. $./corpus -h

**Notes**:

* Requires Python 2.7.
* Requires Numpy.
* Requires Scipy.

###Other Sources:

1. **configuration.py**: contains global script configuration variables
2. **neuralnet.py**: a neural network implementation in numpy
3. **shiftreducer.py**: a shift reduce dependency parser for neural word embeddings
4. **graphparser.py**: a maximum spanning tree dependency parser for neural word embeddings

**Notes**:

* Requires Python 2.7.
* Requires Numpy.
* Requires Scipy.
* You will have to write your own code to use the functions in these files.
