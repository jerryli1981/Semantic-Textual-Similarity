import numpy as np
from utils import *

import collections # for dL

class depTreeRnnModel:

    def __init__(self, relNum, wvecDim):

        self.relNum = relNum
        self.wvecDim = wvecDim

        self.defaultVec = lambda : np.zeros((wvecDim,))

    def initialParams(self, word2vecs, rng):

        # Word vectors
        self.numWords = word2vecs.shape[1]
        self.L = word2vecs[:self.wvecDim, :]

        # Relation layer parameters
        self.WR = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.relNum, self.wvecDim, self.wvecDim)
                    )

        # Hidden layer parameters 
        self.WV = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.b = np.zeros((self.wvecDim))

        self.stack = [self.L, self.WR, self.WV, self.b]


        #create with default_factory : defaultVec = lambda : np.zeros((wvecDim,))
        #make the defaultdict useful for building a dictionary of np array
        #this is very good to save memory, However, this may not support for multiprocess due to pickle error

        #defaultVec = lambda : np.zeros((wvecDim,))
        #dL = collections.defaultdict(defaultVec)
        # Gradients

        # this is important to share memory, notice each epoch need clear this memory
        self.dWR = np.empty((self.relNum, self.wvecDim, self.wvecDim))
        self.dWV = np.empty((self.wvecDim, self.wvecDim))
        self.db = np.empty(self.wvecDim)

    def clearDerivativeSharedMemory(self):

        self.dWR[:] = 0 
        self.dWV[:] = 0
        self.db[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        self.dstack = [self.dL, self.dWR, self.dWV, self.db]

    def forwardProp(self, tree):

        #because many training epoch. 
        tree.resetFinished()

        to_do = []
        to_do.append(tree.root)

        while to_do:
        
            curr = to_do.pop(0)
            curr.vec = self.L[:, curr.index]

            # node is leaf
            if len(curr.kids) == 0:

                # activation function is the normalized tanh
                #curr.hAct = norm_tanh(np.dot(curr.vec, self.WV) + self.b)
                curr.hAct = np.tanh(np.dot(curr.vec, self.WV) + self.b)

                curr.finished=True

            else:

                #check if all kids are finished
                all_done = True
                for index, rel in curr.kids:
                    node = tree.nodes[index]
                    if not node.finished:
                        to_do.append(node)
                        all_done = False

                if all_done:

                    sum = np.zeros(self.wvecDim)
                    for i, rel in curr.kids:
                        W_rel = self.WR[rel.index] # d * d
                        sum += np.dot(tree.nodes[i].hAct, W_rel) 

                    #curr.hAct = norm_tanh(sum + np.dot(curr.vec, self.WV) + self.b)
                    curr.hAct = np.tanh(sum + np.dot(curr.vec, self.WV) + self.b)
        
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        return tree.root.hAct


    def backProp(self, tree, deltas):

        to_do = []
        to_do.append(tree.root)

        tree.root.deltas = deltas

        while to_do:

            curr = to_do.pop(0)
            if len(curr.kids) == 0:
       
                curr.deltas *= derivative_tanh(curr.hAct)

                self.dWV += np.outer(curr.deltas, curr.vec)
                self.db += curr.deltas
                #self.dL[:, curr.index] += np.dot(curr.deltas, self.WV)
                self.dL[curr.index] += np.dot(curr.deltas, self.WV)

            else:

                curr.deltas *= derivative_tanh(curr.hAct)

                self.dWV += np.outer(curr.deltas, curr.vec)
                self.db += curr.deltas
                #self.dL[:, curr.index] += np.dot(curr.deltas, self.WV)
                self.dL[curr.index] += np.dot(curr.deltas, self.WV)

                for i, rel in curr.kids:

                    kid = tree.nodes[i]
                    to_do.append(kid)

                    self.dWR[rel.index] += np.outer(curr.deltas, kid.hAct)

                    rel_vec = self.WR[rel.index]
                    kid.deltas = np.dot(rel_vec.T, curr.deltas)

