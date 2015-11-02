import numpy as np

from utils import *

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
                # j is node id
                x_j = curr.vec
              
                # hj is zero
                i_j = sigmoid( np.dot(self.Wi, x_j) + self.bi)

                # due to k is zero
                f_jk = sigmoid( np.dot(self.Wf, x_j) + self.bf)

                o_j = sigmoid( np.dot(self.Wo, x_j) + self.bo)

                u_j = np.tanh( np.dot(self.Wu, x_j) + self.bu)

                c_j = i_j * u_j

                curr.c_j = c_j
                curr.h_j= o_j * np.tanh(c_j)
                
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

                    x_j = curr.vec

                    h_j_hat = np.zeros((self.wvecDim))

                    sum_f_jk_C_k = np.zeros((self.wvecDim))

                    for i, rel in curr.kids:
                        h_j_hat += tree.nodes[i].h_j 

                        f_jk = sigmoid( np.dot(self.Wf, x_j) + np.dot(self.Uf, tree.nodes[i].h_j )+ self.bf)
                        sum_f_jk_C_k += f_jk * tree.nodes[i].c_j

                    i_j = sigmoid( np.dot(self.Wi, x_j) + np.dot(self.Ui, h_j_hat)+ self.bi)
                    o_j = sigmoid( np.dot(self.Wo, x_j) + np.dot(self.Uo, h_j_hat)+ self.bo)
                    u_j = np.tanh( np.dot(self.Wu, x_j) + np.dot(self.Uu, h_j_hat)+ self.bu)

                    c_j = i_j * u_j + sum_f_jk_C_k

                    curr.c_j = c_j

                    curr.h_j= o_j * np.tanh(c_j)

                    curr.h_j_hat = h_j_hat
                    
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        return tree.root.h_j


    def backProp(self, tree, deltas):

        to_do = []
        to_do.append(tree.root)

        tree.root.deltas = deltas

        while to_do:

            curr = to_do.pop(0)

            if len(curr.kids) == 0:

                x_j = curr.vec

                delta_h_j = curr.deltas
                delta_o_j = delta_h_j
                delta_o_j *= derivative_sigmoid(x_j)
                

                self.dWo += np.outer(delta_o_j, x_j)
                self.dbo += delta_o_j
                self.dL[:, curr.index] += np.dot(self.Wo, delta_o_j)


                delta_c_j = delta_h_j.dot(derivative_tanh(curr.h_j))

                delta_i_j = delta_c_j
                delta_i_j *= derivative_sigmoid(x_j)
                self.dWi += np.outer(delta_i_j, x_j)
                self.dbi += delta_i_j
                self.dL[:, curr.index] += np.dot(self.Wi, delta_i_j)


                delta_u_j = delta_c_j
                delta_u_j *= derivative_sigmoid(x_j)
                self.dWu += np.outer(delta_u_j, x_j)
                self.dbu += delta_u_j
                self.dL[:, curr.index] += np.dot(self.Wu, delta_u_j)


            else:

                x_j = curr.vec

                delta_h_j = curr.deltas
                delta_o_j = delta_h_j
                delta_o_j_1 = delta_o_j * derivative_sigmoid(x_j)
                delta_o_j_2 = delta_o_j * derivative_sigmoid(curr.h_j_hat)
                
                self.dWo += np.outer(delta_o_j_1, x_j)
                self.dUo += np.outer(delta_o_j_2, curr.h_j_hat)
                self.dbo += delta_o_j_1
                self.dL[:, curr.index] += np.dot(self.Wo, delta_o_j_1)


                delta_c_j = delta_h_j * derivative_tanh(curr.h_j)

                delta_i_j = delta_c_j
                delta_i_j_1 = delta_i_j * derivative_sigmoid(x_j)
                delta_i_j_2 = delta_i_j * derivative_sigmoid(curr.h_j_hat)

                self.dWi += np.outer(delta_i_j_1, x_j)
                self.dUi += np.outer(delta_i_j_2, curr.h_j_hat)
                self.dbi += delta_i_j_1
                self.dL[:, curr.index] += np.dot(self.Wi, delta_i_j_1)


                delta_u_j = delta_c_j
                delta_u_j_1 = delta_u_j * derivative_sigmoid(x_j)
                delta_u_j_2 = delta_u_j * derivative_sigmoid(curr.h_j_hat)

                self.dWu += np.outer(delta_u_j_1, x_j)
                self.dUu += np.outer(delta_u_j_2, curr.h_j_hat)
                self.dbu += delta_u_j_1
                self.dL[:, curr.index] += np.dot(self.Wu, delta_u_j_1)

                delta_f_jk = delta_c_j
                delta_f_jk_1 = delta_f_jk * derivative_sigmoid(x_j)
                

                self.dWf += np.outer(delta_f_jk_1, x_j)
                self.dbf += delta_f_jk_1

                for i, rel in curr.kids:

                    kid = tree.nodes[i]
                    to_do.append(kid)

                    delta_f_jk_2 = delta_f_jk * derivative_sigmoid(kid.h_j)
                    self.dUf += np.outer(delta_f_jk_2, kid.h_j)

                    kid.deltas = np.dot(self.Uf, delta_c_j)

