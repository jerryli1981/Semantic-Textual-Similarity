import jpype
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

class ParserError(Exception):
    def __init__(self, *args, **margs):
        Exception.__init__(self, *args,**margs)
           
stanford_parser_home = None

def startJvm():
    import os
    os.environ.setdefault("STANFORD_PARSER_HOME", "./stanford-parser-2011-09-14")
    global stanford_parser_home
    stanford_parser_home = os.environ["STANFORD_PARSER_HOME"]
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/stanford-parser.jar" % stanford_parser_home)
    
startJvm() # one jvm per python instance.

class Parser:

    def __init__(self):

        self.pcfg_model_fname = "%s/grammar/englishPCFG.ser.gz" % stanford_parser_home            
      
        self.parser = jpype.JPackage("edu.stanford.nlp.parser.lexparser").LexicalizedParser(self.pcfg_model_fname)
        
        self.package = jpype.JPackage("edu.stanford.nlp")
        tokenizerFactoryClass = self.package.process.__getattribute__("PTBTokenizer$PTBTokenizerFactory")
        self.tokenizerFactory = tokenizerFactoryClass.newPTBTokenizerFactory(True, True)
           
        self.parser.setOptionFlags(["-retainTmpSubcategories"])
        self.parser.setOptionFlags(["-outputFormat", "wordsAndTags"])
        
        self.stringReader = jpype.JClass("java.io.StringReader")
        
        self.TreebankLanguagePack = jpype.JClass("java.io.StringReader")
       
        self.PennTreebankLanguagePack = jpype.JClass("edu.stanford.nlp.trees.PennTreebankLanguagePack")
        self.GrammaticalStructureFactory = self.PennTreebankLanguagePack().grammaticalStructureFactory()
              
    
    def split_relation(self, text):
        rel_split = text.split('(')
        rel = rel_split[0]
        deps = rel_split[1][:-1]
        if len(rel_split) != 2:
            print 'error ', rel_split
            sys.exit(0)
    
        else:
            dep_split = deps.split(',')
    
            # more than one comma (e.g. 75,000-19)
            if len(dep_split) > 2:
    
                fixed = []
                half = ''
                for piece in dep_split:
                    piece = piece.strip()
                    if '-' not in piece:
                        half += piece
    
                    else:
                        fixed.append(half + piece)
                        half = ''
    
                print 'fixed: ', fixed
                dep_split = fixed
    
            final_deps = []
            for dep in dep_split:
                words = dep.split('-')
                word = words[0]
                ind = int(words[len(words) - 1])
    
                if len(words) > 2:
                    word = '-'.join([w for w in words[:-1]])
    
                final_deps.append( (ind, word.strip()) )
    
            return rel, final_deps

    
    def parseSentence(self, sentence):
    
        sr = self.stringReader(sentence)    
        tokens = self.tokenizerFactory.getTokenizer(sr).tokenize();
        parse = self.parser.apply(tokens)
        if not parse:
            raise ParserError("Could not parse " + sentence)
     
        tokens = []
        pos = []
        lemmas = []
        for token in parse.taggedYield():
            lemma = wordnet_lemmatizer.lemmatize(token.word())
            lemmas.append(lemma)
            tokens.append(token.word())
            pos.append(token.tag())
       
        gs = self.GrammaticalStructureFactory.newGrammaticalStructure(parse)
        
        deps = gs.typedDependenciesCollapsedTree()
        rels = []
        for line in deps:

            rel, final_deps = self.split_relation(str(line))

            govIdx = final_deps[0][0]-1
            depIdx = final_deps[1][0]-1
            rels.append([rel, govIdx, depIdx])
            
        datum  = {"deps_ccTree":rels, "tokens":tokens, "pos":pos, "lemmas":lemmas}
      
        return datum
          
        
           
if __name__ == '__main__':
    
    parser = Parser()
    #parser.parseSentence("A group of kids is playing in a yard and an old man is standing in the background")
    parser.parseSentence("A group of boys in a yard is playing and a man is standing in the background")

