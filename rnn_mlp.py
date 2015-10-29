import numpy as np
import theano
import theano.tensor as T

from scipy.stats import pearsonr

def sigmoid(x):
    """ Sigmoid function """
    x = 1/(1+np.exp(-x))    
    return x

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    f = f*(1-f)    
    return f


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]

class HiddenLayer(object):
    def __init__(self, rng, input_1, input_2, n_in, n_out, W_1=None, W_2=None, b=None,
                 activation=T.nnet.sigmoid):

        if W_1 is None:
            W_1_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_1_values *= 4

            W_1 = theano.shared(value=W_1_values, name='W_1', borrow=True)

        if W_2 is None:
            W_2_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_2_values *= 4

            W_2 = theano.shared(value=W_2_values, name='W_2', borrow=True)


        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W_1 = W_1
        self.W_2 = W_2
        self.b = b

        """
        lin_output = T.dot(input, self.W) + self.b
        """
        lin_output = T.dot(input_1, self.W_1) + T.dot(input_2, self.W_2) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )


        # parameters of the model
        self.params = [self.W_1, self.W_2, self.b]

class MLP(object):

    def __init__(self, rng, input_1, input_2, n_in, n_hidden, n_out):

        self.numLabels = n_out

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input_1=input_1,
            input_2=input_2,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.sigmoid
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W_1).sum() + abs(self.hiddenLayer.W_2).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W_1 ** 2).sum() + (self.hiddenLayer.W_2 ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.output = self.logRegressionLayer.p_y_given_x

    def kl_divergence(self, y):
        newshape=(T.shape(self.output)[1],)
        x = T.reshape(self.output, newshape)
        return T.dot(y, (T.log(y) - T.log(x)).T) / self.numLabels


    def predict_p(self, x_1, x_2):

        W_1_hidden, W_2_hidden, b_hidden = self.hiddenLayer.params

        W_lg, b_lg = self.logRegressionLayer.params

        output = T.nnet.sigmoid( T.dot(x_1, W_1_hidden) + T.dot(x_2, W_2_hidden) + b_hidden)

        p_y_given_x = T.nnet.softmax(T.dot(output, W_lg) + b_lg)

        return p_y_given_x
   

class depTreeRnnModel:

    def __init__(self, relNum, wvecDim, outputDim):

        self.relNum = relNum
        self.wvecDim = wvecDim
        self.outputDim = outputDim


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
                curr.hAct= np.tanh(np.dot(self.WV,curr.vec) + self.b)
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

                    curr.hAct = np.tanh(sum + self.WV.dot(curr.vec) + self.b)
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
                curr.deltas *= (1-curr.hAct**2)

                self.dWV += np.outer(curr.deltas, curr.vec)
                self.db += curr.deltas

                for i, rel in curr.kids:

                    kid = tree.nodes[i]
                    to_do.append(kid)

                    self.dWR[rel.index] += np.outer(curr.deltas, kid.hAct)

                    rel_vec = self.WR[rel.index]
                    kid.deltas = np.dot(rel_vec.T, curr.deltas)

class depTreeLSTMModel:

    def __init__(self, wvecDim, outputDim):

        self.wvecDim = wvecDim
        self.outputDim = outputDim


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



class SGD:

    def __init__(self, rep_model, rng, alpha=0.01, optimizer='sgd', epsilon = 1e-16):

        self.rep_model = rep_model
        self.rep_model.initialGrads()

        self.learning_rate = alpha # learning rate


        x_1 = T.fvector('x_1')  # the data is presented as one sentence output
        x_2 = T.fvector('x_2')  # the data is presented as one sentence output
        y = T.fvector('y')  # the target distribution

        # construct the MLP class
        self.classifier = MLP(rng=rng,input_1=x_1, input_2=x_2, n_in=self.rep_model.wvecDim,
                            n_hidden=50,n_out=self.rep_model.outputDim)

   
        self.mlp_forward = theano.function([x_1,x_2], self.classifier.predict_p(x_1,x_2), allow_input_downcast=True)
        
        L1_reg=0.00
        L2_reg=0.0001 

        cost = self.classifier.kl_divergence(y) + L1_reg * self.classifier.L1+ L2_reg * self.classifier.L2_sqr
        #cost = classifier.kl_divergence(y)
        #cost_function = theano.function([x,y], cost, allow_input_downcast=True)

        hidden_layer_W_1 = self.classifier.hiddenLayer.params[0]
        hidden_layer_W_2 = self.classifier.hiddenLayer.params[1]
        hidden_layer_b = self.classifier.hiddenLayer.params[2]

        deltas_1 = T.dot(hidden_layer_W_1, T.grad(cost,hidden_layer_b))
        deltas_2 = T.dot(hidden_layer_W_2, T.grad(cost,hidden_layer_b))

        #grad_function = theano.function([x,y], T.grad(cost,hidden_layer_b), allow_input_downcast=True)
        self.deltas_function_1 = theano.function([x_1,x_2,y], deltas_1, allow_input_downcast=True)
        self.deltas_function_2 = theano.function([x_1,x_2,y], deltas_2, allow_input_downcast=True)

        gparams = [T.grad(cost, param) for param in self.classifier.params]
        self.accu_grad = theano.function(inputs=[x_1, x_2, y], outputs=gparams,allow_input_downcast=True)


        """
        updates = [ (param, param - self.learning_rate * gparam)
                    for param, gparam in zip(self.classifier.params, gparams)
                  ]

        #self.update_params_mlp = theano.function(inputs=[x, y], outputs=cost, updates=updates,allow_input_downcast=True)
        """
      
        gparam_hw_1 = T.fmatrix('gparam_hw_1')
        updates_hw_1 = [(self.classifier.params[0], self.classifier.params[0] - self.learning_rate * gparam_hw_1)]

        gparam_hw_2 = T.fmatrix('gparam_hw_2')
        updates_hw_2 = [(self.classifier.params[1], self.classifier.params[1] - self.learning_rate * gparam_hw_2)]

        gparam_hb = T.fvector('gparam_hb')
        updates_hb = [(self.classifier.params[2], self.classifier.params[2] - self.learning_rate * gparam_hb)]

        gparam_lw = T.fmatrix('gparam_lw')
        updates_lw = [(self.classifier.params[3], self.classifier.params[3] - self.learning_rate * gparam_lw)]

        gparam_lb = T.fvector('gparam_lb')
        updates_lb = [(self.classifier.params[4], self.classifier.params[4] - self.learning_rate * gparam_lb)]

        self.update_hw_1_mlp = theano.function(inputs=[gparam_hw_1], updates=updates_hw_1, allow_input_downcast=True)
        self.update_hw_2_mlp = theano.function(inputs=[gparam_hw_2], updates=updates_hw_2, allow_input_downcast=True)
        self.update_hb_mlp = theano.function(inputs=[gparam_hb], updates=updates_hb, allow_input_downcast=True)
        self.update_lw_mlp = theano.function(inputs=[gparam_lw], updates=updates_lw, allow_input_downcast=True)
        self.update_lb_mlp = theano.function(inputs=[gparam_lb], updates=updates_lb, allow_input_downcast=True)


        gparam_hw_1_d = T.fmatrix('gparam_hw_1_d')
        updates_hw_1_d = [(self.classifier.params[0], self.classifier.params[0] + gparam_hw_1_d)]

        gparam_hw_2_d = T.fmatrix('gparam_hw_2_d')
        updates_hw_2_d = [(self.classifier.params[1], self.classifier.params[1] + gparam_hw_2_d)]

        gparam_hb_d = T.fvector('gparam_hb_d')
        updates_hb_d = [(self.classifier.params[2], self.classifier.params[2] + gparam_hb_d)]

        gparam_lw_d = T.fmatrix('gparam_lw_d')
        updates_lw_d = [(self.classifier.params[3], self.classifier.params[3] + gparam_lw_d)]

        gparam_lb_d = T.fvector('gparam_lb_d')
        updates_lb_d = [(self.classifier.params[4], self.classifier.params[4] + gparam_lb_d)]


        self.update_hw_1_mlp_d = theano.function(inputs=[gparam_hw_1_d], updates=updates_hw_1_d, allow_input_downcast=True)
        self.update_hw_2_mlp_d = theano.function(inputs=[gparam_hw_2_d], updates=updates_hw_2_d, allow_input_downcast=True)
        self.update_hb_mlp_d = theano.function(inputs=[gparam_hb_d], updates=updates_hb_d, allow_input_downcast=True)
        self.update_lw_mlp_d = theano.function(inputs=[gparam_lw_d], updates=updates_lw_d, allow_input_downcast=True)
        self.update_lb_mlp_d = theano.function(inputs=[gparam_lb_d], updates=updates_lb_d, allow_input_downcast=True)


        self.optimizer = optimizer

        if self.optimizer == 'sgd':
            print "Using sgd.."

        elif self.optimizer == 'adagrad':
            print "Using adagrad..."

            self.gradt_mlp = [epsilon + np.zeros(W.shape.eval()) for W in self.classifier.params]

            self.gradt_rnn = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]

        elif self.optimizer =="adadelta":
            print "Using adadelta..."

            self.gradt_mlp_1 = [epsilon + np.zeros(W.shape.eval()) for W in self.classifier.params]
            self.gradt_mlp_2 = [epsilon + np.zeros(W.shape.eval()) for W in self.classifier.params]

            self.gradt_rnn_1 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]
            self.gradt_rnn_2 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]


    def run(self, trainData, batchSize, epsilon = 1e-16):

        np.random.shuffle(trainData)

        batches = [trainData[idx : idx + batchSize] for idx in xrange(0, len(trainData), batchSize)]

        targetData = np.zeros((len(trainData), self.rep_model.outputDim+1))

        for i, (score, item) in enumerate(trainData):
            sim = score
            ceil = np.ceil(sim)
            floor = np.floor(sim)
            if ceil == floor:
                targetData[i, floor] = 1
            else:
                targetData[i, floor] = ceil-sim 
                targetData[i, ceil] = sim-floor

        targetData = targetData[:, 1:]

        for index, batchData in enumerate(batches):

            targets = targetData[index * batchSize: (index + 1) * batchSize]

            d_hidden_w_1 = np.zeros((self.classifier.hiddenLayer.params[0].shape.eval()))
            d_hidden_w_2 = np.zeros((self.classifier.hiddenLayer.params[1].shape.eval()))
            d_hidden_b = np.zeros((self.classifier.hiddenLayer.params[2].shape.eval()))
            d_log_w = np.zeros((self.classifier.logRegressionLayer.params[0].shape.eval()))
            d_log_b = np.zeros((self.classifier.logRegressionLayer.params[1].shape.eval()))

            d_mlp = [d_hidden_w_1, d_hidden_w_2, d_hidden_b, d_log_w, d_log_b]

            for i, (score, item) in enumerate(batchData): 

                td = targets[i]
                td += epsilon

                first_tree_rep = self.rep_model.forwardProp(item[0])
                second_tree_rep = self.rep_model.forwardProp(item[1])
                mul_rep = first_tree_rep * second_tree_rep
                sub_rep = np.abs(first_tree_rep-second_tree_rep)
                #merged_tree_rep = np.concatenate((first_tree_rep, second_tree_rep))
                #merged_tree_rep = np.concatenate((sub_rep, sub_rep))
                #merged_tree_rep = mul_rep
                #merged_tree_rep = sub_rep


                #due to hidden_layer_b_grad equal delta up, so based on error propogation
                #deltas = self.deltas_function(merged_tree_rep, td) # (n_hidden,)
                deltas_1 = self.deltas_function_1(mul_rep, sub_rep, td)
                deltas_2 = self.deltas_function_2(mul_rep, sub_rep, td)

                self.rep_model.backProp(item[0], deltas_1)
                self.rep_model.backProp(item[0], deltas_2)
                self.rep_model.backProp(item[1], deltas_1)
                self.rep_model.backProp(item[1], deltas_2)
                

                [dhw_1, dhw_2, dhb, dlw, dlb] = self.accu_grad(mul_rep, sub_rep, td)
                d_hidden_w_1 += dhw_1
                d_hidden_w_2 += dhw_2
                d_hidden_b += dhb
                d_log_w += dlw
                d_log_b += dlb

                #example_loss = self.update_params_mlp(merged_tree_rep, td)
                
                #assert np.abs(example_loss-example_loss_2)< 0.00001, "Shit"

                #loss += example_loss

            #loss = loss / batchSize

            #loss += 0.5*(np.sum(self.WV**2) + np.sum(self.WR**2) + optimizer.classifier.L2_sqr)

            for score, item in batchData:
                for tree in item:          
                    tree.resetFinished()

            #begin to update rnn parameters
            if self.optimizer == 'sgd':

                #begin to update mlp parameters
                self.update_hw_1_mlp(d_mlp[0])
                self.update_hw_2_mlp(d_mlp[1])
                self.update_hb_mlp(d_mlp[2])
                self.update_lw_mlp(d_mlp[3])
                self.update_lb_mlp(d_mlp[4])

                #begin to update rnn parameters
                update = self.rep_model.dstack

                self.rep_model.stack[1:] = [P-self.learning_rate*dP for P,dP in zip(self.rep_model.stack[1:],update[1:])]

                # handle dictionary update sparsely
                dL = update[0]
                for j in range(self.rep_model.numWords):
                    self.rep_model.L[:,j] -= self.learning_rate*dL[:,j]

            elif self.optimizer == 'adagrad':

                self.adagrad_mlp(d_mlp)

                self.adagrad_rnn(self.rep_model.dstack)

            elif self.optimizer == 'adadelta':

                self.adadelta_mlp(d_mlp)

                self.adadelta_rnn(self.rep_model.dstack)

        return self.rep_model, self.mlp_forward

    def adagrad_mlp(self, grad):

        self.gradt_mlp = [gt+g**2 for gt,g in zip(self.gradt_mlp, grad)]

        update =  [g*(1./np.sqrt(gt)) for gt,g in zip(self.gradt_mlp,grad)]

        self.update_hw_1_mlp(update[0])
        self.update_hw_2_mlp(update[1])
        self.update_hb_mlp(update[2])
        self.update_lw_mlp(update[3])
        self.update_lb_mlp(update[4])

    def adadelta_mlp(self, grad, eps=1e-6, rho=0.95):

        #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
        self.gradt_mlp_1 = [rho*gt+(1.0-rho)*(g**2) for gt,g in zip(self.gradt_mlp_1, grad)]
 
        #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
        dparam = [ -(np.sqrt(gt2+eps) / np.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(self.gradt_mlp_1, self.gradt_mlp_2, grad)]

        self.update_hw_1_mlp_d(dparam[0])
        self.update_hw_2_mlp_d(dparam[1])
        self.update_hb_mlp_d(dparam[2])
        self.update_lw_mlp_d(dparam[3])
        self.update_lb_mlp_d(dparam[4])

        #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
        self.gradt_mlp_2 = [rho*dt + (1.0-rho)*(d ** 2) for dt, d in zip(self.gradt_mlp_2, dparam)]

    def adagrad_rnn(self, grad):

        # trace = trace+grad.^2
        self.gradt_rnn[1:] = [gt+g**2 for gt,g in zip(self.gradt_rnn[1:],grad[1:])]
        # update = grad.*trace.^(-1/2)
        dparam =  [g*(1./np.sqrt(gt)) for gt,g in zip(self.gradt_rnn[1:],grad[1:])]

        self.rep_model.stack[1:] = [P-self.learning_rate*dP for P,dP in zip(self.rep_model.stack[1:],dparam)]

        # handle dictionary separately
        dL = grad[0]
        dLt = self.gradt_rnn[0]
        for j in range(self.rep_model.numWords):
            dLt[:,j] = dLt[:,j] + dL[:,j]**2
            dL[:,j] = dL[:,j] * (1./np.sqrt(dLt[:,j]))

        # handle dictionary update sparsely
        for j in range(self.rep_model.numWords):
            self.rep_model.L[:,j] -= self.learning_rate*dL[:,j]

    def adadelta_rnn(self, grad, eps=1e-6, rho=0.95):

        #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
        self.gradt_rnn_1[1:] = [rho*gt+(1.0-rho) * (g ** 2) for gt,g in zip(self.gradt_rnn_1[1:], grad[1:])]

        #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
        dparam = [ - (np.sqrt(gt2+eps) / np.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(self.gradt_rnn_1[1:], self.gradt_rnn_2[1:], grad[1:])]
        
        self.rep_model.stack[1:] = [P+ dP for P,dP in zip(self.rep_model.stack[1:],dparam)]

        dL = grad[0]
        dLt_1 = self.gradt_rnn_1[0]
        dLt_2 = self.gradt_rnn_2[0]

        for j in range(self.rep_model.numWords):
            dLt_1[:,j] = rho*dLt_1[:,j]+(1.0-rho)*(dL[:,j] ** 2)
            dL[:,j] = -( np.sqrt(dLt_2[:,j]+eps)/ np.sqrt(dLt_1[:,j]+eps) ) * dL[:,j]

            #update
            dLt_2[:,j] = rho*dLt_2[:,j] + (1.0-rho)*(dL[:,j] ** 2)

        for j in range(self.rep_model.numWords):
            self.rep_model.L[:,j] += dL[:,j]

        #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
        self.gradt_rnn_2[1:] = [rho*dt + (1.0-rho)*( d** 2) for dt, d in zip(self.gradt_rnn_2[1:], dparam)]


def predict(trees, rnn, mlp_forward, epsilon=1e-16):

    # get target distribution for batch
    targets = np.zeros((len(trees), rnn.outputDim+1))
    # compute target distribution 

    for i, (score, item) in enumerate(trees):
        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            targets[i, floor] = 1
        else:
            targets[i, floor] = ceil-sim
            targets[i, ceil] = sim-floor

    targets = targets[:, 1:]

    cost = 0
    corrects = []
    guesses = []

    for i, (score, item) in enumerate(trees):

        td = targets[i]
        td += epsilon

        log_td = np.log(td)

        if len(item) == 1:
            tree_rep= rnn.forwardProp(item[0])
            pd = mlp_forward(tree_rep)

        elif len(item) == 2:
            first_tree_rep= rnn.forwardProp(item[0])
            second_tree_rep = rnn.forwardProp(item[1])
            mul_rep = first_tree_rep * second_tree_rep
            sub_rep = np.abs(first_tree_rep-second_tree_rep)
            #merged_tree_rep = np.concatenate((first_tree_rep, second_tree_rep))
            #merged_tree_rep = mul_rep
            #merged_tree_rep = sub_rep
            pd = mlp_forward(mul_rep,sub_rep)

        predictScore = pd.reshape(rnn.outputDim).dot(np.array([1,2,3,4,5]))

        predictScore = float("{0:.2f}".format(predictScore))

        loss = np.dot(td, log_td-np.log(pd.reshape(rnn.outputDim))) / rnn.outputDim

        cost += loss
        corrects.append(score)
        guesses.append(predictScore)

    #print corrects[:10]
    #print guesses[:10]

    for score, item in trees:
        for tree in item:          
            tree.resetFinished()

    return cost/len(trees), pearsonr(corrects,guesses)[0]


if __name__ == '__main__':

    import dependency_tree as tr     
    trainTrees = tr.loadTrees("train")
    print "train number %d"%len(trainTrees)
     
    numW = len(tr.loadWordMap())

    relMap = tr.loadRelMap()
    relNum = len(relMap)
    word2vecs = tr.loadWord2VecMap()

    wvecDim = 100
    outputDim = 5

    rng = np.random.RandomState(1234)

    #rnn = depTreeRnnModel(relNum, wvecDim, outputDim)
    rnn = depTreeLSTMModel(wvecDim, outputDim)

    rnn.initialParams(word2vecs, rng=rng)
    minibatch = 200

    optimizer = SGD(rep_model=rnn, rng=rng, alpha=0.01, optimizer='adadelta')

    devTrees = tr.loadTrees("dev")

    best_dev_score  = 0.

    print "training model"
    for e in range(1000):
        
        #print "Running epoch %d"%e
        rnn, mlp = optimizer.run(trainTrees, minibatch)
        #print "Time per epoch : %f"%(end-start)
        
        cost, dev_score = predict(devTrees, rnn, mlp)
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print "iter:%d cost: %f dev_score: %f best_dev_score %f"%(e, cost, dev_score, best_dev_score)
        else:
            print "iter:%d cost: %f dev_score: %f"%(e, cost, dev_score)





