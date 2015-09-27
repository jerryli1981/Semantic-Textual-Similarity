import cPickle
import numpy as np
import sys
import re
from collections import defaultdict

from dependency_tree import make_tree 
import networkx as nx

def get_idx_from_sent(sent, word_idx_map, max_l=35, filter_h=6):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

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
           
def build_mergetree_features(revs, word_idx_map):
                
    rel_dict = defaultdict(float) 

    for index, datum in enumerate(revs):
        if index % 100 == 0:
            print index   

        first_parse_output = datum["parse"][0] 
        second_parse_output = datum["parse"][1]   
                
        try:     

            linearizedSentByTree, xs, tree = mergeDepTreesIntoTree(first_parse_output, second_parse_output)

        except:
            print "merge dependency trees into tree has problem"
            continue

        for node in tree.get_nodes():             
            for ind, rel in node.kids:
                if rel not in rel_dict:
                    if rel != "root":
                        rel_dict[rel] += 1
                        
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]
                except:
                    node.ind = 0
            else:
                node.ind = 0
                        
        datum["tree"] = tree

    return rel_dict
      
                 
if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='This is a script for deep learning')

    parser.add_argument('-m', metavar='--mode', 
                        help='feature_engineering, training', 
                        required = True)    
      
    parser.add_argument('-i', metavar='--inputFilePath',
                        help='input file generated from preprocessing.py', 
                        required = True)
    
    parser.add_argument('-c', metavar='--crossvalidation', 
                        help='the number of folds for cross validation', 
                        required = False)

    parser.add_argument('-n', metavar='--numepochs',
                        help='the number of epochs for training', 
                        required = False)
      
    parser.add_argument('-s', metavar='--dimension',
                        help='dimension', 
                        required = False)
    
    parser.add_argument('-t', metavar='--type',
                        help='feature engineering type,'+
                        'merge-into-tree, merge-into-graph', 
                        required = False)
    
    parser.add_argument('-d', metavar='--debug',
                        help='set debug mode, value is debug',
                        required = False)

    args= parser.parse_args()
    if args.d == 'debug':
        import pdb
        pdb.set_trace()

    mode = args.m
    fn = args.i
       
    if mode=="feature_engineering":
        
        feature_engineering_type = args.t
        if feature_engineering_type == None:
            print "feature_engineering_type can't be None"
            sys.exit()
             
        print "loading data...",
        with open(fn,'rb') as f:    
            x = cPickle.load(f)
            revs, word_embedding_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3]
            print "data loaded!, Begin to generate features"
        
        if feature_engineering_type == "merge-into-tree":
            rel_dict = build_mergetree_features(revs, word_idx_map)
                                   
        print "Begin to dump features"
        with open(fn+".features", "wb") as f:
            cPickle.dump([revs, word_embedding_matrix, rel_dict], f)
        print "features is dumped"    
        
    elif mode=="training":
        
        from rnn import evaluate_DT_RNN
        if args.c == None:
            print "crossvalidation can't be None"
            sys.exit()
                
        if args.n == None:
            print "-n, the number of epochs for training can't be None"
            sys.exit()
                
        num_folds = int(args.c)
        numepochs = int(args.n)

        with open(fn+".features", "rb") as f:
            X = cPickle.load(f)

        revs = X[0]
        word_embedding_matrix = X[1]

        partitions = []       
        print "Begin to generate partitons"
        for i in range(num_folds):
            partition = []    
            for rev in revs:   
                if rev["split"]==i:            
                    partition.append('Valid')        
                else:  
                    partition.append('Train')
            partitions.append(partition)

        valid_results = []    

        for i in range(num_folds):
            partition = partitions[i];
            valid_perf, test_perf, params = evaluate_DT_RNN(revs, 
                                                    partition, 
                                                    word_embedding_matrix, 
                                                    rel_dict,
                                                    batch_size = 100,
                                                    n_labels = num_labels,
                                                    n_epochs=numepochs,
                                                    d = dim)
            
            print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
            test_results.append(test_perf)  
            valid_results.append(valid_perf)
            if(test_perf > best_test_perf):
                print "saving RNN model"
                cPickle.dump( ( params, vocab, rel_dict, dim), open('RNN_model_tree_'+fname, 'wb'))
                best_test_perf = test_perf  

                
        print('Average valid performance %f %%' %(np.mean(valid_results) * 100.))        
        print('Average test performance %f %%' %(np.mean(test_results) * 100.))    
        print('Best test performance %f %%' %(best_test_perf * 100.))     
      
    