import theano.tensor as T
import theano
import numpy as np


#these are for mlp_myself
# LogSoftmax is defined as f_i(x) = log(1/a exp(x_i)), where a = sum_j exp(x_j).
def logsoftmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)
    return np.log(x)

def sigmoid(x):
    """ Sigmoid function """
    x = 1/(1+np.exp(-x))    
    return x

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
   

class mlp_by_myself(object):

    def __init__(self):
             #Sigmoid weights
        #self.Wsg = 0.01*np.random.randn(self.sim_nhidden, self.wvecDim)
        self.Wsg = np.random.rand(self.sim_nhidden, self.wvecDim) * 2 * r - r

        # Softmax weights
        #self.Wsm = 0.01*np.random.randn(self.outputDim,self.sim_nhidden) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.Wsm = np.random.rand(self.outputDim, self.sim_nhidden) * 2 * r - r

        self.bsm = np.zeros((self.outputDim))

    def initialGrads(self):

        self.dWsg = np.zeros((self.sim_nhidden, self.wvecDim))
        self.dWsm = np.zeros((self.outputDim, self.sim_nhidden))
        self.dbsm = np.zeros(self.outputDim)

    def forwardProp():

        # compute target distribution  
        td = np.zeros(self.outputDim+1) 
        sim = tree.score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            td[floor] = 1
        else:
            td[floor] = ceil-sim
            td[ceil] = sim-floor

        td = td[1:]
        td += 1e-8
        #compute similarity
        tree.root.sigAct = sigmoid(np.dot(self.Wsg, tree.root.hAct)) #(sim_nhidden,)
        pd= logsoftmax((np.dot(self.Wsm, tree.root.sigAct) + self.bsm).reshape(1, self.outputDim)) #(1, outputDim)
        #KL divergence loss, loss(x, target) = \sum(target_i * (log(target_i) - x_i))
        log_td = np.log(td)
        log_td[log_td == -inf] = 0
        cost = np.dot(td, log_td-pd.reshape(self.outputDim)) / self.outputDim

        tree.root.pd = np.exp(pd.reshape(self.outputDim))
        tree.root.td = td

        if test:
            predictScore = np.exp(pd.reshape(self.outputDim)).dot(np.array([1,2,3,4,5]))
            return cost, float("{0:.2f}".format(predictScore)), tree.score
        else:
            return cost

    def backwardProp():

        norm = -1.0/self.outputDim
        sim_grad = norm * tree.root.td
        sim_grad[sim_grad == -0.] = 0
        
        #softmax gradient
        deltas_sm = sim_grad * (tree.root.pd * (1-tree.root.pd))

        self.dWsm += np.outer(deltas_sm,tree.root.sigAct)
        self.dbsm += deltas_sm

        deltas_sg = np.dot(self.Wsm.T, deltas_sm) #(n_hidden)
  
        
        #f = f*(1-f)   

        deltas_sg *= tree.root.sigAct*(1-tree.root.sigAct)

        self.dWsg += np.outer(deltas_sg, tree.root.hAct)

        tree.root.deltas = np.dot(self.Wsg.T,deltas_sg)




