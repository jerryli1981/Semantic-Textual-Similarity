import numpy as np

from mlp import sigmoid

class depTreeLSTMModel:

    def __init__(self, wvecDim):

        self.wvecDim = wvecDim

    def initialParams(self, word2vecs, rng):

        # Word vectors
        self.numWords = word2vecs.shape[1]
        self.L = word2vecs[:self.wvecDim, :]

        self.Wi = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.Ui = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.bi = np.zeros((self.wvecDim))

        self.Wf = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.Uf = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.bf = np.zeros((self.wvecDim))

        self.Wo = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.Uo = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.bo = np.zeros((self.wvecDim))

        self.Wu = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.Uu = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.bu = np.zeros((self.wvecDim))


        self.stack = [self.L, self.Wi, self.Ui, self.bi, self.Wf, self.Uf, self.bf,
                              self.Wo, self.Uo, self.bo, self.Wu, self.Uu, self.bu ]

    def initialGrads(self):

        #create with default_factory : defaultVec = lambda : np.zeros((wvecDim,))
        #make the defaultdict useful for building a dictionary of np array
        #this is very good to save memory, However, this may not support for multiprocess due to pickle error

        #defaultVec = lambda : np.zeros((wvecDim,))
        #dL = collections.defaultdict(defaultVec)
        self.dL = np.zeros((self.wvecDim, self.numWords))
        self.dWi = np.zeros((self.wvecDim, self.wvecDim))
        self.dUi = np.zeros((self.wvecDim, self.wvecDim))
        self.dbi = np.zeros(self.wvecDim)

        self.dWf = np.zeros((self.wvecDim, self.wvecDim))
        self.dUf = np.zeros((self.wvecDim, self.wvecDim))
        self.dbf = np.zeros(self.wvecDim)

        self.dWo = np.zeros((self.wvecDim, self.wvecDim))
        self.dUo = np.zeros((self.wvecDim, self.wvecDim))
        self.dbo = np.zeros(self.wvecDim)

        self.dWu = np.zeros((self.wvecDim, self.wvecDim))
        self.dUu = np.zeros((self.wvecDim, self.wvecDim))
        self.dbu = np.zeros(self.wvecDim)

        self.dstack = [self.dL, self.dWi, self.dUi, self.dbi, self.dWf, self.dUf, self.dbf,
                                self.dWo, self.dUo, self.dbo, self.dWu, self.dUu, self.dbu ]
        

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

                i_t = sigmoid( np.dot(self.Wi, curr.vec) + self.bi)
                f_t = sigmoid( np.dot(self.Wf, curr.vec) + self.bf)
                o_t = sigmoid( np.dot(self.Wo, curr.vec) + self.bo)
                u_t = np.tanh( np.dot(self.Wu, curr.vec) + self.bu)
                c_t = i_t * u_t

                curr.hAct= o_t * np.tanh(c_t)
                curr.c_t = c_t

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

                    sum_2 = np.zeros((self.wvecDim))
                    for i, rel in curr.kids:
                        sum += tree.nodes[i].hAct 

                        f_k = sigmoid( np.dot(self.Wf, curr.vec) + np.dot(self.Uf, tree.nodes[i].hAct )+ self.bf)
                        sum_2 += f_k * tree.nodes[i].c_t

                    i_t = sigmoid( np.dot(self.Wi, curr.vec) + np.dot(self.Ui, sum)+ self.bi)
                    o_t = sigmoid( np.dot(self.Wo, curr.vec) + np.dot(self.Uo, sum)+ self.bo)
                    u_t = np.tanh( np.dot(self.Wu, curr.vec) + np.dot(self.Uu, sum)+ self.bu)

                    c_t = i_t * u_t + sum_2

                    curr.hAct= o_t * np.tanh(c_t)
                    curr.c_t = c_t

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

                deltas_o_j = curr.deltas
                deltas_c_j = curr.deltas*(1-curr.hAct**2)

                self.dWo += np.outer(deltas_o_j, curr.vec)
                self.dUo += np.outer(deltas_o_j, curr.hAct)
                self.dbo += deltas_o_j


                self.dWi += np.outer(deltas_c_j, curr.vec)
                self.dUi += np.outer(deltas_c_j, curr.hAct)
                self.dbi += deltas_c_j

                self.dWo += np.outer(deltas_c_j, curr.vec)
                self.dUo += np.outer(deltas_c_j, curr.hAct)
                self.dbo += deltas_c_j

                for i, rel in curr.kids:

                    kid = tree.nodes[i]
                    to_do.append(kid)

                    self.dWf += np.outer(deltas_c_j, kid.vec)
                    self.dUf += np.outer(deltas_c_j, kid.hAct)
                    self.dbf += deltas_c_j

                    kid.deltas = deltas_c_j

