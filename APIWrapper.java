import java.util.Collection;
import java.util.*;
import java.io.StringReader;
import java.io.IOException;
import java.lang.Exception;

import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;

class APIWrapper {
	
	private static String parserModel;
	private static String filename;
	private static Tree parse;

	public static void usage() {
		//TODO- complete the usage 
		System.out.println( "Usage: java APIWrapper <text_file>");
	}

	public static void main(String[] args) {
		parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
		if (args.length<1) {
			usage();
			return;
		}
		else 
			filename = args[0];

		LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);
		TreebankLanguagePack tlp = lp.treebankLanguagePack();
		GrammaticalStructureFactory gsf = null;

		if (tlp.supportsGrammaticalStructures()) {
      		gsf = tlp.grammaticalStructureFactory();
    	}
    	for (List<HasWord> sentence : new DocumentPreprocessor(filename)) {
    		//System.out.println(sentence);
      		try {
      			parse = lp.apply(sentence);
      			//parse.pennPrint();
      			//System.out.println();
      			if (gsf != null) {
        			GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        			Collection<TypedDependency> tdl = gs.typedDependencies();//CCprocessed();
        			for (TypedDependency el:tdl) {
        				System.out.println(el.toString());
        			}
       				//System.out.println(tdl);
        			System.out.println();
        		}
        	} catch (Exception e) {
        			e.printStackTrace();
        	}
		}
	}
}