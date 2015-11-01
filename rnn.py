import numpy as np
from utils import *

class depTreeRnnModel:

    def __init__(self, relNum, wvecDim, activation):

        self.relNum = relNum
        self.wvecDim = wvecDim
        self.activation = activation

    def initialParams_anotherway():
        
        #generate the same random number
        np.random.seed(12341)

        r = np.sqrt(6)/np.sqrt(201)

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.numWords = word2vecs.shape[1]
        # scale by 0.01 can pass gradient check
        self.L = word2vecs[:self.wvecDim, :]

        # Relation layer parameters
        #self.WR = 0.01*np.random.randn(self.relNum, self.wvecDim, self.wvecDim)
        self.WR = np.random.rand(self.relNum, self.wvecDim, self.wvecDim) * 2 * r -r

        # Hidden layer parameters 
        #self.WV = 0.01*np.random.randn(self.wvecDim, self.wvecDim)
        self.WV = np.random.rand(self.wvecDim, self.wvecDim) * 2 * r - r
        self.b = np.zeros((self.wvecDim))


        self.stack = [self.L, self.WR, self.WV, self.b]


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

    def initialGrads(self):

        #create with default_factory : defaultVec = lambda : np.zeros((wvecDim,))
        #make the defaultdict useful for building a dictionary of np array
        #this is very good to save memory, However, this may not support for multiprocess due to pickle error

        #defaultVec = lambda : np.zeros((wvecDim,))
        #dL = collections.defaultdict(defaultVec)
        self.dL = np.zeros((self.wvecDim, self.numWords))
        self.dWR = np.zeros((self.relNum, self.wvecDim, self.wvecDim))
        self.dWV = np.zeros((self.wvecDim, self.wvecDim))
        self.db = np.zeros(self.wvecDim)

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
                if self.activation == "tanh":
                    curr.hAct = np.tanh(np.dot(self.WV,curr.vec) + self.b)
                elif self.activation == "sigmoid":
                    curr.hAct = sigmoid(np.dot(self.WV,curr.vec) + self.b)
                else:
                    raise "incorrect activaton function"
                
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

                    sum = np.zeros((self.wvecDim))
                    for i, rel in curr.kids:
                        rel_vec = self.WR[rel.index]
                        sum += rel_vec.dot(tree.nodes[i].hAct) 

                    if self.activation == "tanh":        
                        curr.hAct = np.tanh(sum + self.WV.dot(curr.vec) + self.b)
                    elif self.activation == "sigmoid":
                        curr.hAct = sigmoid(sum + self.WV.dot(curr.vec) + self.b)    
                    else:
                        raise "incorrect activation function"
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

                self.dL[:, curr.index] += curr.deltas

            else:

                # derivative of tanh
                if self.activation == "tanh":
                    curr.deltas *= derivative_tanh(curr.hAct)
                elif self.activation == "sigmoid":
                    curr.deltas *= derivative_sigmoid(curr.hAct)
                else:
                    raise "incorrect activation function"

                self.dWV += np.outer(curr.deltas, curr.vec)
                self.db += curr.deltas

                for i, rel in curr.kids:

                    kid = tree.nodes[i]
                    to_do.append(kid)

                    self.dWR[rel.index] += np.outer(curr.deltas, kid.hAct)

                    rel_vec = self.WR[rel.index]
                    kid.deltas = np.dot(rel_vec.T, curr.deltas)

