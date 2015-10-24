import numpy as np
import random # for random shuffle train data
from numpy import inf

import theano
import theano.tensor as T

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

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

#T.tanh
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):

        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.numLabels = n_out

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.output = self.logRegressionLayer.p_y_given_x

    def kl_divergence(self, y):
        newshape=(T.shape(self.output)[1],)
        x = T.reshape(self.output, newshape)
        return T.dot(y, (T.log(y) - T.log(x)).T) / self.numLabels

class depTreeRnnModel:

    def __init__(self, relNum, wvecDim, outputDim):

        self.relNum = relNum
        self.wvecDim = wvecDim
        self.outputDim = outputDim


    def initialParams(self, word2vecs):

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

    def forwardProp(self, tree, test=False):

        #because many training epoch. 
        tree.resetFinished()

        cost  =  0.0 

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
                        pass
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


    #Minbatch stochastic gradient descent
    def train(self, trainData, alpha, batchSize, numProcess=None):

        #random.shuffle(trainData)

        batches = [trainData[x : x + batchSize] for x in xrange(0, len(trainData), batchSize)]

        x = T.fvector('x')  # the data is presented as one sentence output
        y = T.fvector('y')  # the target distribution

        rng = np.random.RandomState(1234)

        # construct the MLP class
        classifier = MLP(
            rng=rng,
            input=x,
            n_in=self.wvecDim,
            n_hidden=50,
            n_out=self.outputDim
        )

        L1_reg=0.00 
        L2_reg=0.0001

        epsilon = 1e-8
      
        #mlp_forward = theano.function([x], classifier.output, allow_input_downcast=True)

        cost = classifier.kl_divergence(y) + L1_reg * classifier.L1+ L2_reg * classifier.L2_sqr

        cost_function = theano.function([x,y], cost, allow_input_downcast=True)

        gparams = [T.grad(cost, param) for param in classifier.params]

        W = classifier.hiddenLayer.params[0]

        grad_function = theano.function([x,y], T.grad(cost,W), allow_input_downcast=True)

        for batchData in batches:

            # get target distribution for batch
            targets = np.zeros((batchSize, self.outputDim+1))
            # compute target distribution 
    
            for i, tree in enumerate(batchData):
                sim = tree.score
                ceil = np.ceil(sim)
                floor = np.floor(sim)
                if ceil == floor:
                    targets[i, floor] = 1
                else:
                    targets[i, floor] = ceil-sim
                    targets[i, ceil] = sim-floor

            targets = targets[:, 1:]

            loss = 0.

            for i, tree in enumerate(batchData): 

                tree_rep = self.forwardProp(tree)


                """
                norm = -1.0/self.outputDim
                sim_grad = norm * tree.root.td
                sim_grad[sim_grad == -0.] = 0

                deltas_sm = sim_grad

                deltas_sg = np.dot(Wsm.T, deltas_sm)
  
                deltas_sg *= tree.root.sigAct*(1-tree.root.sigAct)
              
                deltas = np.dot(self.Wsg.T,deltas_sg)
                """

                grad = grad_function(tree_rep,td) # d * n_hidden
                deltas = T.dot(W.T,gw)

                self.backProp(tree, deltas)

                td = targets[i]
                td += epsilon

                """
                output = mlp_forward(tree_rep)
                log_td = np.log(td)

                example_loss = np.dot(td, log_td-np.log(output.reshape(self.outputDim))) / self.outputDim
                """

                example_loss = cost_function(tree_rep,td)
                

                #assert np.abs(example_loss-example_loss_2)< 0.00001, "Shit"

                """
                norm = -1.0/self.outputDim
                sim_grad = norm * td
                sim_grad[sim_grad == -0.] = 0
                """

                loss += example_loss

        for tree in trainData:
                tree.resetFinished()


if __name__ == '__main__':

    import dependency_tree as treeM      
    train = treeM.loadTrees()
    print "train number %d"%len(train)
     
    numW = len(treeM.loadWordMap())

    relMap = treeM.loadRelMap()
    relNum = len(relMap)
    word2vecs = treeM.loadWord2VecMap()

    wvecDim = 12
    outputDim = 5

    rnn = depTreeRnnModel(relNum, wvecDim, outputDim)
    rnn.initialParams(word2vecs)
    minibatch = 10
    
    rnn.train(train, 1e-2, minibatch)

