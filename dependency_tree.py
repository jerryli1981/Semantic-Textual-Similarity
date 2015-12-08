from nltk.corpus import wordnet as wn

class Relation:
    def __init__(self, mention, index):
        self.mention = mention
        self.index = index

class Node:

    def __init__(self, word, index):
        self.word = word
        self.kids = []
        self.index = index
        self.finished = False

class DTree:

    def __init__(self, toks, parents, labels,vocab,rel_vocab):

        self.nodes = []
        for tok in toks:
            self.nodes.append(Node(tok, vocab[tok])) 

        self.dependencies = []

        for i, (govIdx, rel) in enumerate(zip(parents, labels)):
            depIdx = i + 1
            govIdx = int(govIdx)
            if govIdx > 0:
                self.nodes[govIdx-1].kids.append((depIdx, Relation(rel,rel_vocab[rel])))

            self.dependencies.append((govIdx, depIdx))

    def resetFinished(self):
        for node in self.nodes:
            node.finished = False

    def mergeWith(self, dtree):
        
        for govIdx_d, depIdx_d in dtree.dependencies:
            
            if govIdx_d == 0:
                continue
            
            govTok_d = dtree.nodes[govIdx_d-1].word
            depTok_d = dtree.nodes[depIdx_d-1].word

            rel_d = None

            for idx, rel in dtree.nodes[govIdx_d-1].kids:
                if idx == depIdx_d:
                    rel_d = rel

            assert rel_d != None
         
            add = False
   
            for govIdx, depIdx in self.dependencies:

                if govIdx == 0:
                    continue 
       
                govTok = self.nodes[govIdx-1].word           
                depTok = self.nodes[depIdx-1].word
                
                govMatch = isMatch(govTok, govTok_d)
                depMatch = isMatch(depTok, depTok_d)
                
                if govMatch and not depMatch:
                    add = True
                    matchedTextGovToken = govTok
                    matchedTextGovIdx = govIdx
                
                if govMatch and depMatch:
                    add = False
                    break
                         
            if add:
                depGloIdx_d = dtree.nodes[depIdx_d-1].index
                cNode = Node(depTok_d,depGloIdx_d)
                self.nodes.append(cNode)
                newDepIdx = len(self.nodes)
                self.dependencies.append((matchedTextGovIdx, newDepIdx))
                self.nodes[matchedTextGovIdx].kids.append((newDepIdx, rel_d))
  
def isMatch(T, H, synonym = False, entail = False, antonym = False, hypernym = False, hyponym = False):
    
    
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



