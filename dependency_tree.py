UNK = 'UNK'

import networkx as nx

class Node:

    def __init__(self, word):
        self.word = word
        self.kids = []
        self.parent = []
        self.index = None

class DTree:

    def __init__(self, parse_result, label=None, score=None):

        self.deps = parse_result["deps_ccTree"]

        #use lemmas instead of toks 
        self.lemmas = parse_result["lemmas"]
            
        self.label = label
        self.score = score

        self.root = self.make_tree(self.deps, self.lemmas)

    def make_tree(self, deps, lemmas):        
        # store tree as adjacent list
        self.nodes = []
        for tok in lemmas:
            self.nodes.append(Node(tok)) 
        
        # add dependency edges between nodes
        rootIdx = None
        for rel, govIdx, depIdx in deps:
            if govIdx == -1:
                rootIdx = depIdx
                continue

            pNode = Node(lemmas[govIdx])
            cNode = Node(lemmas[depIdx])

            self.nodes[govIdx].kids.append((cNode, rel))
            self.nodes[depIdx].parent.append((pNode, rel))

        return self.nodes[rootIdx]

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

        for rel_d, govIdx_d, depIdx_d in dtree.deps:
            
            if govIdx_d == -1:
                continue
            else:
                govLemma_d = dtree.lemmas[govIdx_d]


            depLemma_d = dtree.lemmas[depIdx_d]
         
            add = False

            for rel, govIdx, depIdx in self.deps:

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
                pNode = self.nodes[matchedTextGovIdx]   
                self.nodes[matchedTextGovIdx].kids.append((cNode, rel_d))
                cNode.parent.append((pNode, rel_d))
                                                               
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

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """
    from collections import defaultdict
    import cPickle as pickle
    file = 'training_dataset'
    print "Reading dataset to build word map.."
    trees = []
    with open(file,'r') as fid:
        dataset = pickle.load(fid)
        for index, datum in enumerate(dataset):
            if index %1000 == 0 :
                print index
            first_parse, second_parse = datum["parse"]
            score = datum["score"]
            first_depTree = DTree(first_parse, score = score)
            second_depTree = DTree(second_parse,score = score)
            mergedTree, isDag = first_depTree.mergeWith(second_depTree)
            if not isDag:
                print "merge is not dag"
                continue
            trees.append(mergedTree)

    print "Counting words to give each word an index.."
    
    words = defaultdict(int)
    for tree in trees:
        for node in tree.nodes:
            words[node.word] += 1

    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word

    print "Saving wordMap to wordMap.bin"
    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

def loadTrees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    import cPickle as pickle
    wordMap = loadWordMap()
    file = 'training_dataset'
    print "Loading training_dataset"
    trees = []
    with open(file,'r') as fid:
        dataset = pickle.load(fid)
        for index, datum in enumerate(dataset):
            if index %1000 == 0 :
                print index
            first_parse, second_parse = datum["parse"]
            score = datum["score"]
            first_depTree = DTree(first_parse, score = score)
            second_depTree = DTree(second_parse,score = score)
            mergedTree, isDag = first_depTree.mergeWith(second_depTree)
            if not isDag:
                print "merge is not dag"
                continue
            trees.append(mergedTree)

    for tree in trees:
        for node in tree.nodes:
            if node.word not in wordMap:
                node.index = wordMap[UNK]
            else:
                node.index = wordMap[node.word]

    return trees

if __name__=='__main__':

    _Debug_ = False
    if _Debug_:
        import pdb
        pdb.set_trace()

    buildWordMap()
    #train = loadTrees() 