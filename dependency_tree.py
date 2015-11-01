UNK = 'UNK'

import networkx as nx
import copy
import numpy as np

class Relation:
    def __init__(self, mention):
        self.mention = mention
        self.index = None

class Node:

    def __init__(self, word):
        self.word = word
        self.kids = []
        self.parent = []
        self.index = None
        self.finished = False

class DTree:

    def __init__(self, parse_result, label=None, score=None):

        self.deps = parse_result["deps_ccTree"]

        #use lemmas instead of toks 
        self.lemmas = parse_result["lemmas"]
            
        self.label = label
        self.score = score

        # store tree as adjacent list
        self.nodes = []
        for tok in self.lemmas:
            self.nodes.append(Node(tok)) 
        
        # add dependency edges between nodes
        rootIdx = None
        dependencies = []
        for rel, govIdx, depIdx in self.deps:
            if govIdx == -1:
                rootIdx = depIdx
                continue
            self.nodes[govIdx].kids.append((depIdx, Relation(rel)))
            self.nodes[depIdx].parent.append((govIdx, Relation(rel)))
            dependencies.append((govIdx, depIdx))

        self.root = self.nodes[rootIdx]

        G = nx.DiGraph()
        G.add_edges_from(dependencies)

        self.is_dag = nx.is_directed_acyclic_graph(G)

    def resetFinished(self):
        for node in self.nodes:
            node.finished = False


    def mergeWith(self, dtree):
        
        #for cycle detection
        dependencies = []
        for rel, govIdx, depIdx in self.deps:
             
            """
            if text_govIdx == -1:
                continue 
            else:
                govLemma = self.lemmas[govIdx]
                govToken = self.tokens[govIdx]
                         
            depLemma = self.lemmas[depIdx])
            depToken = self.tokens[depIdx])
            """

            dependencies.append((govIdx, depIdx))
        
        # here deep copy is necessary due to later we will update self.deps
        deps = copy.deepcopy(self.deps)

        for rel_d, govIdx_d, depIdx_d in dtree.deps:
            
            if govIdx_d == -1:
                continue
            else:
                govLemma_d = dtree.lemmas[govIdx_d]


            depLemma_d = dtree.lemmas[depIdx_d]
         
            add = False
   
            for rel, govIdx, depIdx in deps:

                if govIdx == -1:
                    continue 
                else:
                    govLemma = self.lemmas[govIdx]
        
                             
                depLemma = self.lemmas[depIdx]
                
                govMatch = isMatch(govLemma, govLemma_d)
                depMatch = isMatch(depLemma, depLemma_d)
                
                if govMatch and not depMatch:
                    add = True
                    matchedTextGovToken = govLemma
                    matchedTextGovIdx = govIdx
                
                if govMatch and depMatch:
                    add = False
                    break
                         
            if add:
                cNode = Node(depLemma_d)
                self.nodes.append(cNode)
                newDepIdx = len(self.nodes)-1
                dependencies.append((matchedTextGovIdx, newDepIdx))
                self.nodes[matchedTextGovIdx].kids.append((newDepIdx, Relation(rel_d)))
                self.nodes[newDepIdx].parent.append((matchedTextGovIdx, Relation(rel_d)))

                #update this for get_rel
                self.deps.append((rel_d, matchedTextGovIdx, newDepIdx))

                                                               
        G = nx.DiGraph()
        G.add_edges_from(dependencies)
        is_dag = nx.is_directed_acyclic_graph(G)

        return self, is_dag

    
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




def loadWordMap():
    import cPickle as pickle
    
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def loadRelMap():
    import cPickle as pickle
    
    with open('relMap.bin','r') as fid:
        return pickle.load(fid)

def loadWord2VecMap():
    import cPickle as pickle
    
    with open('word2vec.bin','r') as fid:
        return pickle.load(fid)


def buildWordRelMap(train=None, dev=None, test=None):
    """
    Builds map of all words in training set
    to integer values.
    """
    from collections import defaultdict
    import cPickle as pickle
    print "Reading dataset to build word map.."
    trees = []
    if train != None:
        with open(train,'r') as fid:
            dataset = pickle.load(fid)
            for index, datum in enumerate(dataset):
                if index %1000 == 0 :
                    print index
                first_parse, second_parse = datum["parse"]
                score = float(datum["score"])
                first_depTree = DTree(first_parse, score = score)
                second_depTree = DTree(second_parse,score = score)
                trees.append((first_depTree, second_depTree))
    if dev != None:
        with open(dev,'r') as fid:
            dataset = pickle.load(fid)
            for index, datum in enumerate(dataset):
                if index %1000 == 0 :
                    print index
                first_parse, second_parse = datum["parse"]
                score = float(datum["score"])
                first_depTree = DTree(first_parse, score = score)
                second_depTree = DTree(second_parse,score = score)
                trees.append((first_depTree, second_depTree))

    if test != None:
        with open(test,'r') as fid:
            dataset = pickle.load(fid)
            for index, datum in enumerate(dataset):
                if index %1000 == 0 :
                    print index
                first_parse, second_parse = datum["parse"]
                score = float(datum["score"])
                first_depTree = DTree(first_parse, score = score)
                second_depTree = DTree(second_parse,score = score)
                trees.append((first_depTree, second_depTree))
    
    print "Counting words to give each word an index.."
    
    words = defaultdict(int)
    rels = defaultdict(int)
    for first_tree, second_tree in trees:

        for node in first_tree.nodes:
            words[node.word] += 1
        for rel, gov, dep in first_tree.deps:
            #reduce total number relations
            rel = rel.split("_")[0]
            rels[rel] += 1

        for node in second_tree.nodes:
            words[node.word] += 1
        for rel, gov, dep in second_tree.deps:
            #reduce total number relations
            rel = rel.split("_")[0]
            rels[rel] += 1


    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word

    relMap = dict(zip(rels.iterkeys(),xrange(len(rels))))

    print "Saving wordMap to wordMap.bin"
    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

    print "Saving relMap to relMap.bin"
    with open('relMap.bin','w') as fid:
        pickle.dump(relMap,fid)


def build_word2Vector_glove():
    print "building word2vec"

    """
    Loads 300x1 word vecs from glove
    """
    import gzip
    vocab = loadWordMap()
    word_vecs = {}
    with gzip.open("glove.6B.300d.txt.gz", "rb") as f:
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


    assert len(vocab) == len(word_vecs), "length of vocab mush equal with word_vecs"   

    
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_embedding_matrix = np.zeros(shape=(k,vocab_size))            

    for i, word in enumerate(word_vecs):
        word_embedding_matrix[:,i] = word_vecs[word]
    #print "len of word_embedding_matrix",len(word_embedding_matrix)
    #print "len of vocab",len(vocab)
    #print "len of word_vecs",len(word_vecs)
    import cPickle as pickle
    print "Saving word2vec to word2vec.bin"
    with open('word2vec.bin','w') as fid:
        pickle.dump(word_embedding_matrix,fid)


def loadTrees(dataSet='train', merged=False):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    import cPickle as pickle
    wordMap = loadWordMap()
    relMap = loadRelMap()
    
    file = dataSet+'_dataset'

    print "Loading %s dataset..."%dataSet
    trees = []
    with open(file,'r') as fid:
        dataset = pickle.load(fid)
        for index, datum in enumerate(dataset):
            #if index %1000 == 0 :
                #print index

            first_parse, second_parse = datum["parse"]

            score = float(datum["score"])
            first_depTree = DTree(first_parse, score = score)
            second_depTree = DTree(second_parse,score = score)
            if merged:
                mergedTree, isDag = first_depTree.mergeWith(second_depTree)
                if not isDag:
                    #print "merge is not dag"
                    continue
                trees.append((score, [mergedTree])) 
            else:
                if first_depTree.is_dag and second_depTree.is_dag:
                    trees.append((score, [first_depTree, second_depTree])) 

    if merged:
        for score, [tree] in trees:
            for node in tree.nodes:
                if node.word not in wordMap:
                    node.index = wordMap[UNK]
                else:
                    node.index = wordMap[node.word]

                if len(node.kids) != 0:
                    for depIdx, rel in node.kids:
                        rel.index = relMap[rel.mention.split("_")[0]]
                if len(node.parent) != 0:
                    for govIdx, rel in node.parent:
                        rel.index = relMap[rel.mention.split("_")[0]]
    else:

        for score, [first_tree, second_tree] in trees:
            for node in first_tree.nodes:
                if node.word not in wordMap:
                    node.index = wordMap[UNK]
                else:
                    node.index = wordMap[node.word]

                if len(node.kids) != 0:
                    for depIdx, rel in node.kids:
                        rel.index = relMap[rel.mention.split("_")[0]]
                if len(node.parent) != 0:
                    for govIdx, rel in node.parent:
                        rel.index = relMap[rel.mention.split("_")[0]]

            for node in second_tree.nodes:
                if node.word not in wordMap:
                    node.index = wordMap[UNK]
                else:
                    node.index = wordMap[node.word]

                if len(node.kids) != 0:
                    for depIdx, rel in node.kids:
                        rel.index = relMap[rel.mention.split("_")[0]]
                if len(node.parent) != 0:
                    for govIdx, rel in node.parent:
                        rel.index = relMap[rel.mention.split("_")[0]]

    return trees


if __name__=='__main__':

    _Debug_ = False
    if _Debug_:
        import pdb
        pdb.set_trace()

    buildWordRelMap("train_dataset","dev_dataset")

    build_word2Vector_glove()