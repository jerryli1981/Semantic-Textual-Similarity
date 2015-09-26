import numpy as np
import re
import gzip
import sys
import cPickle
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf8')

from dependency_tree import make_tree 
import networkx as nx

def get_word_matrix(word_vecs, k=300):
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


def build_word2Vector_glove(fname, vocab):
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

            
def isMatch(T, H, synonym = False, entail = False, antonym = False, hypernym = False
    , hyponym = False):
    from nltk.corpus import wordnet as wn
    
    is_exact_match = False
    if T == H:
        is_exact_match = True

    synsets_T = wn.synsets(T)
    synsets_H = wn.synsets(H)

    is_synonym = False
    is_entail = False
    is_antonmy = False
    is_hypernym = False
    is_hyponym = False
    
    if synonym == True:
        
        lemmas_T = [str(lemma.name()) for ss in synsets_T for lemma in ss.lemmas()]
        lemmas_H = [str(lemma.name()) for ss in synsets_H for lemma in ss.lemmas()]
    
        c = list(set(lemmas_T).intersection(set(lemmas_H)))
    
        if len(c) > 0 or H in lemmas_T:
            is_synonym = True
        else:
            is_synonym = False

    elif entail == True:
        
        for s_T in synsets_T:
            for s_H in synsets_H:
                if s_H in s_T.entailments():
                    is_entail = True 

    elif antonym == True:
       
        nega_T = [str(nega.name()) for ss in synsets_T for lemma in ss.lemmas() for nega in lemma.antonyms()]
        if H in nega_T:
            is_antonmy = True

    elif hypernym == True:
            
        for s_T in synsets_T:
            for s_H in synsets_H:
                
                if s_H in s_T.hyponyms():
                    is_hypernym = True
                            
                if s_T in [synset for path in s_H.hypernym_paths() for synset in path]:
                    is_hypernym = True           

    elif hyponym == True:
      
        for s_T in synsets_T:
            for s_H in synsets_H:
                                       
                if s_T in s_H.hyponyms():
                    is_hyponym = True
                
                if s_H in [synset for path in s_T.hypernym_paths() for synset in path]:
                    is_hyponym = True           
    
    return is_exact_match or is_synonym or is_entail or is_antonmy or is_hypernym or is_hyponym

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
    
    return sent.strip(), xs, depTree
               
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

    
def build_training_data(data_file_path, cv=10):
    from stanford_parser_wrapper import Parser
    import traceback
    parser = Parser()    
    revs = []
    vocab = defaultdict(float)

    with open(data_file_path, "rb") as f:
        
        for index, line in enumerate(f):
            if index % 100 == 0:   
                print index  

            instance = line.strip().split('\t')  
            first_sentence = instance[0].strip()
            second_sentence = instance[1].strip()
                    
            try:
                first_parse_output = parser.parseSentence(first_sentence)
            except:
                print "first_sentence can't be parsing"
                #print first_sentence
                #traceback.print_exc()
                continue
            try:
                second_parse_output = parser.parseSentence(second_sentence)
            except:
                print "second_sentence can't be parsing"
                #print second_sentence
                #traceback.print_exc()
                continue
                                        
            score = instance[2].strip()
                    
            alignments = generateAlignments(first_parse_output, second_parse_output)
            
            try:
                linearizedSentByGraph, graph, isCyc = mergeDepTreesIntoGraph(first_parse_output, 
                                                                         second_parse_output) 
            except:
                print "merge dependency trees into graph has problem"
                continue
            
            if isCyc == False:
                print "graph contain cycle"                
                continue                   
            try:     
                linearizedSentByTree, xs, tree = mergeDepTreesIntoTree(first_parse_output, second_parse_output)
            except:
                print "merge dependency trees into tree has problem"
                continue
                                          
            for word in first_parse_output["tokens"]:
                vocab[word.lower()] += 1
                
            for word in second_parse_output["tokens"]:
                vocab[word.lower()] += 1

            datum  = {"score":score, 
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
            
    return revs, vocab
           
if  __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='This is a script for prepprocessing datasets')
    
    parser.add_argument('-i', metavar='--inputFilePath', 
            			help='input file path', 
                        required = True)
    
    parser.add_argument('-m', metavar ='--mode',
                        help='training or testing', 
                        required = True)
    
    parser.add_argument('-o', metavar='--outputFileName', 
                        help='the output file name', 
                        required = True)
    
    parser.add_argument('-c', metavar='--crossvalidation', 
                        help='the number of folds for cross validation', 
                        required = False)
    
    parser.add_argument('-k', metavar='--mikolov', 
                        help='the path of mikolov word2vector', 
                        required = False)
    
    parser.add_argument('-g', metavar='--glove', 
                        help='the path of glove word2vector', 
                        required = False)
    parser.add_argument('-d', metavar='--debug',
                        help='set debug mode, value is debug',
                        required = False)
    
    args= parser.parse_args()
    if args.d == "debug":
	import pdb
	pdb.set_trace()

    if args.k == None and args.g == None:
        print "word2vec can't be None"
        sys.exit()

    w2v_file_m = args.k
    w2v_file_g = args.g

    data_file_path = args.i
    output_file_path = args.o

    if args.m == "training":
        if args.c == None:
            print "crossvalidation can't be None"
            sys.exit()

        num_folds = int(args.c)
        
        print "Loading data...",        
        revs, vocab = build_training_data(data_file_path, cv=num_folds)
        print "data loaded!"
        
        print "number of instances: " + str(len(revs))
        print "vocab size: " + str(len(vocab))
        
        print "building word2vec from trained word embedding"
        if w2v_file_m != None:
            word_embedding_matrix, word_idx_map = build_word2Vector_mikolov(w2v_file_m, vocab)
        elif w2v_file_g != None:
            word_embedding_matrix, word_idx_map = build_word2Vector_glove(w2v_file_g, vocab) 
        print "word2vec is builded"
    
        cPickle.dump([revs, word_embedding_matrix, word_idx_map, vocab], open(output_file_path, "wb"))
        print "datasets created!"
    
