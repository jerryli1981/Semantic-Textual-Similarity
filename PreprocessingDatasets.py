from nltk.corpus import wordnet as wn
#from nltk.parse.dependencygraph import DependencyGraph
from nltk import tree, treetransforms
from nltk import Tree
import networkx as nx

from DependencyTree import make_tree  

from collections import defaultdict
import numpy as np
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1 # it is make sence because i=0 is zero vector as padding
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def build_word2Vector__glove(fname, vocab):
    import gzip
    """
    Loads 300x1 word vecs from glove
    """
    word_vecs = {}
    with gzip.open(fname, "rb") as f:
        for line in f:
           toks = line.split(' ')
           word = toks[0]
           if word in vocab:
               word_vecs[word] = np.fromiter(toks[1:], dtype='float32')  
               
    k = 300
    min_df = 1
    
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

    
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    word_embedding_matrix = np.zeros(shape=(vocab_size+1, k))            
    word_embedding_matrix[0] = np.zeros(k)
    i = 1 # it is make sence because i=0 is zero vector as padding
    for word in word_vecs:
        word_embedding_matrix[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
        
    return word_embedding_matrix, word_idx_map

def build_word2Vector_mikolov(fname, vocab):
    import gzip
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    k = 300
    min_df = 1
    word_vecs = {}
    with gzip.open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
                
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
            
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    word_embedding_matrix = np.zeros(shape=(vocab_size+1, k))            
    word_embedding_matrix[0] = np.zeros(k)
    i = 1 # it is make sence because i=0 is zero vector as padding
    for word in word_vecs:
        word_embedding_matrix[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
        
    return word_embedding_matrix, word_idx_map


def isSynonyms(T, H):
    
    lemmas_T = [str(lemma.name()) for ss in wn.synsets(T) for lemma in ss.lemmas()]
    
    lemmas_H = [str(lemma.name()) for ss in wn.synsets(H) for lemma in ss.lemmas()]
    
    c = list(set(lemmas_T).intersection(set(lemmas_H)))
    
    if T == H:
        return True
    elif len(c) > 0:
        return True
    elif H in lemmas_T:
        return True
    else:
        return False
    
def isAntonyms(T, H):
    
    nega_T = [str(nega.name()) for ss in wn.synsets(T) for lemma in ss.lemmas() for nega in lemma.antonyms()]
    if H in nega_T:
        return True
    else:
        return False
    
def isHypernmys(T, H):
    synsets_T = wn.synsets(T)
    synsets_H = wn.synsets(H)
    
    for s_T in synsets_T:
        for s_H in synsets_H:
            
            if s_H in s_T.hyponyms():
                return True
                        
            if s_T in [synset for path in s_H.hypernym_paths() for synset in path]:
                return True            
    return False    
    
def isHyponmys(T, H):
    synsets_T = wn.synsets(T)
    synsets_H = wn.synsets(H)
    
    for s_T in synsets_T:
        for s_H in synsets_H:
                                   
            if s_T in s_H.hyponyms():
                return True
            
            if s_H in [synset for path in s_T.hypernym_paths() for synset in path]:
                return True            
    return False    

def isWNEntailments(T, H):
    synsets_T = wn.synsets(T)
    synsets_H = wn.synsets(H)
        
    for s_T in synsets_T:
        for s_H in synsets_H:
            if s_H in s_T.entailments():
                return True
             
    return False  

def isMatch(T, H):
    
    if T == H:
        return True
    
    """
    else:
        return False
    """
    
    #return isSynonyms(T, H) or isHypernmys(T, H) or isHyponmys(T, H) or isWNEntailments(T, H) or isAntonyms(T, H)
    #return isSynonyms(T, H) or isWNEntailments(T, H) or isAntonyms(T, H)
    return isSynonyms(T, H) or isWNEntailments(T, H)

    """
    Tree
    
    Average (SICK.p.4MatchCond )valid performance 73.400000 %
    Average test performance 70.095918 %
    Best test performance 70.571429 %
    
    Average (3Hypon) valid performance 74.638333 %
    Average test performance 72.453061 %
    Best test performance 72.857143 %
    
    Average (3Hyper) valid performance 72.975000 %
    Average test performance 70.663265 %
    Best test performance 71.285714 %
    
    Average (SICK.p.5MatchCond) valid performance 70.275000 %
    Average test performance 68.248980 %
    Best test performance 69.122449 %
    
    Average (SICK.p.3AntMatchCond) valid performance 71.975000 %
    Average test performance 71.277551 %
    Best test performance 71.918367 %
    
    Average (SICK.p.2MatchCond) valid performance 74.825000 %
    Average test performance 73.179592 %
    Best test performance 73.877551 %
    
    Average (SICK.p.1MatchCond) valid performance 74.125000 %
    Average test performance 73.400000 %
    Best test performance 74.020408 %
    *************************************
    Graph
    Average (SICK.p.1MatchCond) valid performance 76.325000 %
    Average test performance 74.891837 %
    Best test performance 75.122449 %
    
    Average (SICK.p.2MatchCond) valid performance 75.825000 %
    Average test performance 74.928571 %
    Best test performance 75.224490 %

    """

def mergeDepTreesIntoGraph(text_parse_output, hypo_parse_output):
    
    text_lemmas = text_parse_output["lemmas"]
    text_tokens = text_parse_output["tokens"]
    text_deps =text_parse_output["deps_ccTree"]
          
    dependencies = []
    for text_pair in text_deps:
        text_rel = str(text_pair[0])
        text_govIdx = text_pair[1]
        if text_govIdx == -1:
            text_govLemma = "ROOT"
            text_govToken = "ROOT" 
        else:
            text_govLemma = str(text_lemmas[text_govIdx])
            text_govToken = str(text_tokens[text_govIdx])
                     
        text_depIdx = text_pair[2]    
        text_depLemma = str(text_lemmas[text_depIdx])
        text_depToken = str(text_tokens[text_depIdx])
               
        relation = text_rel +"("+text_govToken+"-"+str(text_govIdx+1)+", "+text_depToken+"-"+str(text_depIdx+1)+")"
        dependencies.append(relation)
        
  
    hypo_lemmas = hypo_parse_output["lemmas"]
    hypo_tokens = hypo_parse_output["tokens"]
    hypo_deps =hypo_parse_output["deps_ccTree"]
            

    for hypo_pair in hypo_deps:
        hypo_rel = str(hypo_pair[0])
        hypo_govIdx = hypo_pair[1]
        hypo_depIdx = hypo_pair[2]
        hypo_govLemma = None
        hypo_govToken = None
        if hypo_govIdx == -1:
            continue
        else:
            hypo_govLemma = str(hypo_lemmas[hypo_govIdx])
            hypo_govToken = str(hypo_tokens[hypo_govIdx])
        hypo_depLemma = str(hypo_lemmas[hypo_depIdx])
        hypo_depToken = str(hypo_tokens[hypo_depIdx])
       
        addNewOneEdge = False
        addNewTwoEdge = False
            
        for text_pair in text_deps:
            text_rel = str(text_pair[0])
            text_govIdx = text_pair[1]
            text_depIdx = text_pair[2]
            text_govLemma = None
            text_govToken = None
            if text_govIdx == -1:
                continue                
            else:
                text_govLemma = str(text_lemmas[text_govIdx])
                text_govToken = str(text_tokens[text_govIdx])
                
            text_depLemma = str(text_lemmas[text_depIdx])
            text_depToken = str(text_tokens[text_depIdx])
                    
            govMatch = isMatch(text_govLemma, hypo_govLemma)
            depMatch = isMatch(text_depLemma, hypo_depLemma)
                
            if govMatch and depMatch:      
                addNewOneEdge = False
                addNewTwoEdge = False   
                break
            elif govMatch and not depMatch:
                addNewOneEdge = True
                addNewTwoEdge = False 
                matchedTextGovToken = text_govToken
                matchedTextGovIdx = text_govIdx
            elif not govMatch and depMatch:
                
                matchedTextDepToken = text_depToken
                matchedTextDepIdx = text_depIdx
                
                addNewTwoEdge = True
                addNewOneEdge = False
            else:
                addNewOneEdge = True
                addNewTwoEdge = True 
                    
        if addNewOneEdge and not addNewTwoEdge:
        
            text_tokens.append(hypo_depToken)
            new_hypo_depIdx = len(text_tokens)-1
                  
            relation = hypo_rel +"("+matchedTextGovToken+"-"+str(matchedTextGovIdx+1)+", "+hypo_depToken+"-"+str(new_hypo_depIdx+1)+")"
            dependencies.append(relation)
              
        elif addNewTwoEdge and not addNewOneEdge:

       
            text_tokens.append(hypo_govToken)
            new_hypo_govIdx = len(text_tokens)-1
                      
            relation1 = "root" +"(ROOT-0"+", "+hypo_govToken+"-"+str(new_hypo_govIdx+1)+")"
            relation2 = hypo_rel +"("+hypo_govToken+"-"+str(new_hypo_govIdx+1)+", "+matchedTextDepToken+"-"+str(matchedTextDepIdx+1)+")"
            dependencies.append(relation1)
            dependencies.append(relation2)
            
        elif addNewTwoEdge and addNewOneEdge:
            
            text_tokens.append(hypo_govToken)
            new_hypo_govIdx = len(text_tokens)-1
                      
            relation1 = "root" +"(ROOT-0"+", "+hypo_govToken+"-"+str(new_hypo_govIdx+1)+")"
            dependencies.append(relation1)
            
            text_tokens.append(hypo_depToken)
            new_hypo_depIdx = len(text_tokens)-1
            relation2 = hypo_rel +"("+hypo_govToken+"-"+str(new_hypo_govIdx+1)+", "+hypo_depToken+"-"+str(new_hypo_depIdx+1)+")"
            dependencies.append(relation2)
                      
                      
    G = nx.DiGraph()
    for pair in dependencies:    
        eles = re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',pair).groups()
        rel = eles[0]
        gov = eles[1]+"-"+eles[2]
        dep = eles[3]+"-"+eles[4]
        G.add_edge(gov, dep, relation_name=rel)
        
    isCyc = nx.is_directed_acyclic_graph(G)


    linear = list(nx.dfs_preorder_nodes(G, "ROOT-0"))
    toks = [re.sub(r'-(\d+)', "", ele) for ele in linear if ele != "ROOT-0"]
    sent = " ".join(toks)
    
    xs=[re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',r).groups() for r in dependencies]
    relations = [(ele[0], [(int(ele[2]), ele[1]),(int(ele[4]), ele[3])]) for ele in xs]
    graph = make_tree(relations)
    
    return sent.strip(), graph, isCyc

def treeBinarization(tree, factor="right", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="^"):
    if horzMarkov is None: horzMarkov = 999
    nodeList = [(tree, [tree.label()])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node,Tree):

            # parent annotation
            parentString = ""
            originalNode = node.label()
            if vertMarkov != 0 and node != tree and isinstance(node[0],Tree):
                parentString = "%s<%s>" % (parentChar, "::".join(parent))
                node.set_label(node.label() + parentString)
                parent = [originalNode] + parent[:vertMarkov - 1]

            # add children to the agenda before we mess with them
            for child in node:
                nodeList.append((child, parent))

            # chomsky normal form factorization
            if len(node) > 2:
                childNodes = []
                for child in node:
                    if isinstance(child,Tree):
                        childNodes.append(child.label())
                    else:
                        childNodes.append(child)
                nodeCopy = node.copy()
                node[0:] = [] # delete the children

                curNode = node
                numChildren = len(nodeCopy)
                for i in range(1,numChildren - 1):
                    if factor == "right":
                        newHead = "%s%s<%s>%s" % (originalNode, childChar, "::".join(childNodes[i:min([i+horzMarkov,numChildren])]),parentString) # create new head
                        newNode = Tree(newHead, [])
                        curNode[0:] = [nodeCopy.pop(0), newNode]
                    else:
                        newHead = "%s%s<%s>%s" % (originalNode, childChar, "::".join(childNodes[max([numChildren-i-horzMarkov,0]):-i]),parentString)
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]

                    curNode = newNode

                curNode[0:] = [child for child in nodeCopy]


def mergeDepTreesIntoTree(text_parse_output, hypo_parse_output):
   
    text_lemmas = text_parse_output["lemmas"]
    text_poss = text_parse_output["pos"]
    text_tokens = text_parse_output["tokens"]
    text_deps =text_parse_output["deps_ccTree"]
   
    dependencies = []
    for text_pair in text_deps:
        text_rel = str(text_pair[0])
        text_govIdx = text_pair[1]
        if text_govIdx == -1:
            text_govLemma = "ROOT"
            text_govToken = "ROOT" 
        else:
            text_govLemma = str(text_lemmas[text_govIdx])
            text_govToken = str(text_tokens[text_govIdx])
                     
        text_depIdx = text_pair[2]    
        text_depLemma = str(text_lemmas[text_depIdx])
        text_depToken = str(text_tokens[text_depIdx])
               
        relation = text_rel +"("+text_govToken+"-"+str(text_govIdx+1)+", "+text_depToken+"-"+str(text_depIdx+1)+")"
        dependencies.append(relation)
    

    hypo_lemmas = hypo_parse_output["lemmas"]
    hypo_tokens = hypo_parse_output["tokens"]
    hypo_poss = hypo_parse_output["pos"]
    hypo_deps =hypo_parse_output["deps_ccTree"]
    for hypo_pair in hypo_deps:
        hypo_rel = str(hypo_pair[0])
        hypo_govIdx = hypo_pair[1]
        hypo_depIdx = hypo_pair[2]

        if hypo_govIdx == -1:
            continue
        else:
            hypo_govLemma = str(hypo_lemmas[hypo_govIdx])
            hypo_govToken = str(hypo_tokens[hypo_govIdx])
            hypo_govPos = str(hypo_poss[hypo_govIdx])

        hypo_depLemma = str(hypo_lemmas[hypo_depIdx])
        hypo_depToken = str(hypo_tokens[hypo_depIdx])
        hypo_depPos = str(hypo_poss[hypo_depIdx])
          
        add = False
        noExist = True
        for text_pair in text_deps:
            text_rel = str(text_pair[0])
            text_govIdx = text_pair[1]
            if text_govIdx == -1:
                continue         
            else:
                text_govLemma = str(text_lemmas[text_govIdx])
                text_govToken = str(text_tokens[text_govIdx])
                text_govPos = str(text_poss[text_govIdx])
                
            text_depIdx = text_pair[2]
            text_depLemma = str(text_lemmas[text_depIdx])
            text_depToken = str(text_tokens[text_depIdx])
            
            #print (text_govToken, text_depToken)
            #print (hypo_govToken, hypo_depToken)
            govMatch = isMatch(text_govLemma, hypo_govLemma)
            depMatch = isMatch(text_depLemma, hypo_depLemma)
            
            if govMatch and not depMatch:
                noExist = False
                add = True
                #govMatch = isMatch(text_govLemma, hypo_govLemma)
                #depMatch = isMatch(text_depLemma, hypo_depLemma)
                matchedTextGovToken = text_govToken
                matchedTextGovIdx = text_govIdx
            
            if govMatch and depMatch:
                
                #govMatch = isMatch(text_govLemma, hypo_govLemma)
                #depMatch = isMatch(text_depLemma, hypo_depLemma)
                noExist = False
                add = False
                break
            
            if not govMatch and depMatch:
                noExist = False
            
                
        if add == True:
            text_tokens.append(hypo_depToken)
            text_poss.append(hypo_depPos)
            new_hypo_depIdx = len(text_tokens)-1
                  
            relation = hypo_rel +"("+matchedTextGovToken+"-"+str(matchedTextGovIdx+1)+", "+hypo_depToken+"-"+str(new_hypo_depIdx+1)+")"
            dependencies.append(relation)
            
        if noExist == True:
            
            text_tokens.append(hypo_govToken)
            new_hypo_govIdx = len(text_tokens)-1
                      
            relation1 = "root" +"(ROOT-0"+", "+hypo_govToken+"-"+str(new_hypo_govIdx+1)+")"
            dependencies.append(relation1)
            
            text_tokens.append(hypo_depToken)
            new_hypo_depIdx = len(text_tokens)-1
            relation2 = hypo_rel +"("+hypo_govToken+"-"+str(new_hypo_govIdx+1)+", "+hypo_depToken+"-"+str(new_hypo_depIdx+1)+")"
            dependencies.append(relation2)
                      
                                                                
    G = nx.DiGraph()
    for pair in dependencies:    
        eles = re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',pair).groups()
        rel = eles[0]
        gov = eles[1]+"-"+eles[2]
        dep = eles[3]+"-"+eles[4]
        G.add_edge(gov, dep, relation_name=rel)

    linear = list(nx.dfs_preorder_nodes(G, "ROOT-0"))
    toks = [re.sub(r'-(\d+)', "", ele) for ele in linear if ele != "ROOT-0"]
    sent = " ".join(toks)
    
    """
    malt_tab_rep=''
    xs=[re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',r).groups() for r in dependencies] 
    ys= sorted(xs,key=lambda elem:int(elem[4]))
    
    for y in ys:
        malt_tab_rep+="{}\t{}\t{}\t{}\n".format(y[3],text_poss[int(y[4])-1],y[2],y[0])
             
    dg = DependencyGraph(malt_tab_rep)  
    """
    xs=[re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',r).groups() for r in dependencies]
    relations = [(ele[0], [(int(ele[2]), ele[1]),(int(ele[4]), ele[3])]) for ele in xs]
    depTree = make_tree(relations)
    
    #depTree = tree.copy(deep=True)
    #treeBinarization(tr)

    return sent.strip(), xs, depTree
               
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()


def parse_data_StanfordParserWrapper(train_data_file, test_data_file):
    from Stanford_Parser_Wrapper import *
    
    parser = Parser()
    
    
    result = []
    i = 0
    with open(train_data_file, "rb") as f:
        f.readline()
        for line in f:   
            print i
            i += 1   

            instance = line.strip().split('\t')  
            pair_id =  instance[0].strip().lower()
            first_sentence = instance[1].strip().lower()
            second_sentence = instance[2].strip().lower()

            score = instance[3]
            label = instance[4]
            

            text_parse_output = parser.parseSentence(first_sentence)
            hypo_parse_output = parser.parseSentence(second_sentence)

            result.append((text_parse_output, hypo_parse_output))
                             
                                   
    with open(test_data_file, "rb") as f:
        f.readline()
        for line in f:    
            print i
            i += 1 

            instance = line.strip().split('\t') 
            pair_id =  instance[0].strip().lower()  
            first_sentence = instance[1].strip().lower()
            second_sentence = instance[2].strip().lower()

            score = instance[3]
            label = instance[4]
                       

            text_parse_output = parser.parseSentence(first_sentence)
            hypo_parse_output = parser.parseSentence(second_sentence)
            result.append((text_parse_output, hypo_parse_output))
                             
    return result

def parse_data_StanfordCoreNlpWrapper(train_data_file, test_data_file):
    
    from stanford_corenlp_pywrapper import sockwrap
    p=sockwrap.SockWrap('nerparse',corenlp_jars=
                        ["/home/peng/.m2/repository/edu/stanford/nlp/stanford-corenlp/3.4.1/stanford-corenlp-3.4.1.jar",
                         "/home/peng/.m2/repository/edu/stanford/nlp/stanford-corenlp/3.4.1/stanford-corenlp-3.4.1-models.jar"])

    
    result = []
    i = 0
    with open(train_data_file, "rb") as f:
        f.readline()
        for line in f:   
            print i
            i += 1
                        
            instance = line.strip().split('\t')  
            pair_id =  instance[0].strip().lower()
            first_sentence = instance[1].strip().lower()
            second_sentence = instance[2].strip().lower()
            score = instance[3]
            label = instance[4]
            
            text_parse_output = p.parse_doc(first_sentence)["sentences"][0]
            hypo_parse_output = p.parse_doc(second_sentence)["sentences"][0]
            result.append((text_parse_output, hypo_parse_output))
                             
                                   
    with open(test_data_file, "rb") as f:
        f.readline()
        for line in f:    
            print i
            i += 1 
            instance = line.strip().split('\t') 
            pair_id =  instance[0].strip().lower()  
            first_sentence = instance[1].strip().lower()
            second_sentence = instance[2].strip().lower()
            score = instance[3]
            label = instance[4]
                       
            text_parse_output = p.parse_doc(first_sentence)["sentences"][0]
            hypo_parse_output = p.parse_doc(second_sentence)["sentences"][0]
            result.append((text_parse_output, hypo_parse_output))
                             
    return result

def generateAlignments(text_parse_output, hypo_parse_output):
    
    text_lemmas = text_parse_output["lemmas"]
    text_tokens = text_parse_output["tokens"]
    text_deps =text_parse_output["deps_ccTree"]
          
    text_dependencies = []
    for text_pair in text_deps:
        text_rel = str(text_pair[0])
        text_govIdx = text_pair[1]
        if text_govIdx == -1:
            text_govLemma = "ROOT"
            text_govToken = "ROOT" 
        else:
            text_govLemma = str(text_lemmas[text_govIdx])
            text_govToken = str(text_tokens[text_govIdx])
                     
        text_depIdx = text_pair[2]    
        text_depLemma = str(text_lemmas[text_depIdx])
        text_depToken = str(text_tokens[text_depIdx])
               
        relation = text_rel +"("+text_govToken+"-"+str(text_govIdx+1)+", "+text_depToken+"-"+str(text_depIdx+1)+")"
        text_dependencies.append(relation)
        
  
    hypo_lemmas = hypo_parse_output["lemmas"]
    hypo_tokens = hypo_parse_output["tokens"]
    hypo_deps =hypo_parse_output["deps_ccTree"]
            
    
    hypothesis_dependencies = []
    
    for hypo_pair in hypo_deps:
        hypo_rel = str(hypo_pair[0])
        hypo_govIdx = hypo_pair[1]
        if hypo_govIdx == -1:
            hypo_govLemma = "ROOT"
            hypo_govToken = "ROOT" 
        else:
            hypo_govLemma = str(hypo_lemmas[hypo_govIdx])
            hypo_govToken = str(hypo_tokens[hypo_govIdx])
                     
        hypo_depIdx = hypo_pair[2]    
        hypo_depLemma = str(hypo_lemmas[hypo_depIdx])
        hypo_depToken = str(hypo_tokens[hypo_depIdx])
               
        relation = hypo_rel +"("+hypo_govToken+"-"+str(hypo_govIdx+1)+", "+hypo_depToken+"-"+str(hypo_depIdx+1)+")"
        hypothesis_dependencies.append(relation)

    candidate_alignment_pairs = [(tp, hp) for tp in text_dependencies for hp in hypothesis_dependencies] 
    return  candidate_alignment_pairs

    
def build_data(train_data_file, test_data_file, parserDumpFile, cv=10):
    
    parseddatas = cPickle.load(open(parserDumpFile,"rb"))
        
    labelIdxMap = {}
    revs = []
    vocab = defaultdict(float)
    i = 0
    idx = 0
    with open(train_data_file, "rb") as f:
        f.readline()
        for line in f:   
            print i
            i += 1
                                        
            instance = line.strip().split('\t')  
            first_sentence = instance[1].strip().lower()
            second_sentence = instance[2].strip().lower()
            pair_id =  instance[0].strip().lower()
            score = instance[3]
            label = instance[4]
            
            if label not in labelIdxMap:
                labelIdxMap[label] = idx
                idx += 1
            
            labelIdx = labelIdxMap[label]
                       
            text_parse_output, hypo_parse_output = parseddatas[i-1] 
            
            alignments = generateAlignments(text_parse_output, hypo_parse_output)
            
            try:
                linearizedSentByGraph, graph, isCyc = mergeDepTreesIntoGraph(text_parse_output, 
                                                                         hypo_parse_output) 
            except:
                print "Graph linearization has problem"
                continue
            
            if isCyc == False:
                print "graph contain cycle"
                
                continue                    
            linearizedSentByTree, xs, tree = mergeDepTreesIntoTree(text_parse_output, hypo_parse_output)

                                
            text_tokens = text_parse_output["tokens"]          
            for word in text_tokens:
                vocab[word] += 1
                
            hypo_tokens = hypo_parse_output["tokens"]
            for word in hypo_tokens:
                vocab[word] += 1
                
                              
            datum  = {"id":pair_id,
                      "score":score, 
                      "label":labelIdx,
                      "text": first_sentence+"\t"+second_sentence, 
                      "linearbyTree": linearizedSentByTree,   
                      "linearbyGraph": linearizedSentByGraph, 
                      "tree": tree,    
                      "graph": graph,
                      "deps": xs,  
                      "alignments": alignments,                
                      "num_words_merged_Tree": len(linearizedSentByTree.split()),
                      "num_words_merged_Graph": len(linearizedSentByGraph.split()),
                      "num_words_single": max(len(first_sentence.split()),len(second_sentence.split())),
                      "num_subtrees": len(list(tree.get_nodes())),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
            
              
    with open(test_data_file, "rb") as f:
        f.readline()
        for line in f:    
            print i
            i += 1 
            
            instance = line.strip().split('\t')  
            first_sentence = instance[1].strip().lower()
            second_sentence = instance[2].strip().lower()
            pair_id =  instance[0].strip().lower()
            score = instance[3]
            label = instance[4]
            
            if label not in labelIdxMap:
                labelIdxMap[label] = idx
                idx += 1
            
                
            labelIdx = labelIdxMap[label]
            
            text_parse_output, hypo_parse_output = parseddatas[i-1]
            
            alignments = generateAlignments(text_parse_output, hypo_parse_output)
            
            try:
                linearizedSentByGraph, graph, isCyc = mergeDepTreesIntoGraph(text_parse_output, 
                                                                         hypo_parse_output) 
            except:
                print "Graph linearization has problem"
                continue

            
            if isCyc == False:
                print "XXXX "+ pair_id
                continue                       
            linearizedSentByTree, xs, tree = mergeDepTreesIntoTree(text_parse_output, hypo_parse_output)
            
            
            text_tokens = text_parse_output["tokens"]          
            for word in text_tokens:
                vocab[word] += 1
                
            hypo_tokens = hypo_parse_output["tokens"]
            for word in hypo_tokens:
                vocab[word] += 1
              
            datum  = {"id":pair_id,
                      "score":score, 
                      "label":labelIdx,
                      "text": first_sentence+"\t"+second_sentence, 
                      "linearbyTree": linearizedSentByTree,   
                      "linearbyGraph": linearizedSentByGraph,   
                      "tree": tree, 
                      "graph": graph, 
                      "deps": xs, 
                      "alignments": alignments,                      
                      "num_words_merged_Tree": len(linearizedSentByTree.split()),
                      "num_words_merged_Graph": len(linearizedSentByGraph.split()),
                      "num_words_single": max(len(first_sentence.split()),len(second_sentence.split())),
                      "num_subtrees": len(list(tree.get_nodes())),
                      "split": -1}
            revs.append(datum)
            
    return revs, vocab
           
if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='This is a script for text entailment')
    
    parser.add_argument('-i', '--trainFilePath', 
                        help='the train File Path', 
                        required = True)
    
    parser.add_argument('-j', '--testFilePath', 
                        help='the test File Path', 
                        required = True)
    
    parser.add_argument('-m', '--mode', 
                        help='parsing or processing', 
                        required = True)
    
    parser.add_argument('-p', '--parsedFileName', 
                        help='parsing or processing', 
                        required = True)
    
    parser.add_argument('-o', '--outputFileName', 
                        help='the output file name', 
                        required = False)
    
    parser.add_argument('-c', '--crossvalidation', 
                        help='the number of folds for cross validation', 
                        required = False)
    
    parser.add_argument('-k', '--mikolov', 
                        help='the path of mikolov word2vector', 
                        required = False)
    
    parser.add_argument('-g', '--glove', 
                        help='the path of glove word2vector', 
                        required = False)
    
    args= parser.parse_args()
       
    train_data_file = args.trainFilePath
    test_data_file = args.testFilePath
    parserDumpFile = args.parsedFileName
    
    mode = args.mode
    
    if mode == "parsing":
             
        print "Begin to parsing dataset"
        parserResult = parse_data_StanfordParserWrapper(train_data_file, test_data_file)
        print "Begin to dump parser results"
        cPickle.dump(parserResult, open(parserDumpFile, "wb"))

    elif mode == "processing":
        
        if args.outputFileName == None:
            print "output file name can't be None"
            sys.exit() 
        
        datasetName = args.outputFileName
        
        if args.crossvalidation == None:
            print "crossvalidation can't be None"
            sys.exit()

        num_folds = int(args.crossvalidation)
        
        if args.mikolov == None and args.glove == None:
            print "word2vec can't be None"
            sys.exit()
        
        w2v_file_m = args.mikolov
        w2v_file_g = args.glove
        
        
        print "Loading data...",        
        revs, vocab = build_data(train_data_file, test_data_file, parserDumpFile, cv=num_folds)
        print "data loaded!"
        
        print "number of instances: " + str(len(revs))
        print "vocab size: " + str(len(vocab))
        
        print "building word2vec from trained word embedding"
        if w2v_file_m != None:
            word_embedding_matrix, word_idx_map = build_word2Vector_mikolov(w2v_file_m, vocab)
        elif w2v_file_g != None:
            word_embedding_matrix, word_idx_map = build_word2Vector_glove(w2v_file_g, vocab) 
        print "word2vec is builded"
    
        cPickle.dump([revs, word_embedding_matrix, word_idx_map, vocab], open(datasetName, "wb"))
        print "datasets created!"
    