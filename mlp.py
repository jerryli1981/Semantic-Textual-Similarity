import theano.tensor as T
import theano
import numpy as np
from utils import *

class LogisticRegression(object):

    def __init__(self, rng, input, n_in, n_out):

        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )

        self.W = theano.shared(value=W_values, name='W', borrow=True)

        """
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        """

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

        # keep track of model input
        self.input = input
                
    def kl_divergence(self, y):

        return T.sum(y * (T.log(y) - T.log(self.p_y_given_x))) / y.shape[0]

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
            if activation == T.nnet.sigmoid:
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
            if activation == T.nnet.sigmoid:
                W_2_values *= 4

            W_2 = theano.shared(value=W_2_values, name='W_2', borrow=True)


        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W_1 = W_1
        self.W_2 = W_2
        self.b = b

        lin_output = T.dot(input_1, self.W_1) + T.dot(input_2, self.W_2) + self.b

        self.output = activation(lin_output)

        # parameters of the model
        self.params = [self.W_1, self.W_2, self.b]


class MLP(object):

    def __init__(self, rng, input_1, input_2, n_in, n_hidden, n_out, activation):

        self.numLabels = n_out

        if activation == "tanh":
            act_function = T.tanh
        elif activation == "sigmoid":
            act_function = T.nnet.sigmoid

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input_1=input_1,
            input_2=input_2,
            n_in=n_in,
            n_out=n_hidden,
            activation=act_function
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            rng=rng,
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

        self.kl_divergence = self.logRegressionLayer.kl_divergence

        self.output = self.logRegressionLayer.p_y_given_x


class my_mlp(object):

    def __init__(self, rng, n_in, n_hidden, n_out):

        self.numLabels = n_out
        self.rng = rng

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

    def initial_params_twoInput(self):

        self.W_1_hidden = self.rng.uniform(
                    low=-np.sqrt(6. / (self.n_in + self.n_hidden)),
                    high=np.sqrt(6. / (self.n_in + self.n_hidden)),
                    size=(self.n_in, self.n_hidden)
                )

        self.W_2_hidden = self.rng.uniform(
                    low=-np.sqrt(6. / (self.n_in + self.n_hidden)),
                    high=np.sqrt(6. / (self.n_in + self.n_hidden)),
                    size=(self.n_in, self.n_hidden)
                )

        self.b_hidden = np.zeros((self.n_hidden,))

        self.W_output = self.rng.uniform(
                    low=-np.sqrt(6. / (self.n_hidden + self.n_out)),
                    high=np.sqrt(6. / (self.n_hidden + self.n_out)),
                    size=(self.n_hidden, self.n_out)
                )

        self.b_output = np.zeros((self.n_out,))

        self.dW_1_hidden = np.zeros((self.n_in, self.n_hidden))

        self.dW_2_hidden = np.zeros((self.n_in, self.n_hidden))

        self.db_hidden = np.zeros((self.n_hidden,))

        self.dW_output = np.zeros((self.n_hidden, self.n_out))

        self.db_output = np.zeros((self.n_out,))

        self.params = [self.W_1_hidden, self.W_2_hidden, self.b_hidden, self.W_output, self.b_output]

        self.dstack = [self.dW_1_hidden, self.dW_2_hidden, self.db_hidden, self.dW_output, self.db_output]
        
    def forwardProp_2(self, input_1, input_2, td):

        lin_output = np.dot(input_1, self.W_1_hidden) + np.dot(input_2, self.W_2_hidden) + self.b_hidden

        activation = sigmoid(lin_output) #(n_hidden,)

        pd= logsoftmax( (np.dot(activation, self.W_output) + self.b_output).reshape(1, self.numLabels)) #(1, outputDim)
        #KL divergence loss, loss(x, target) = \sum(target_i * (log(target_i) - x_i))
        log_td = np.log(td)

        cost = np.dot(td, log_td - pd.reshape(self.numLabels) )

        pd = np.exp(pd.reshape(self.numLabels))

        return cost, activation, pd

    def backwardProp_2(self, input_1, input_2, activation, pd, td):
        
        
        norm = -1.0/self.numLabels
        deltas_kld = norm * td
        deltas_kld[deltas_kld == -0.] = 0
        

        #softmax gradient
        #deltas_logistic = deltas_kld * sigmoid_grad(pd)
        #softmax error. this is correct based on the consider softmax and cross entropy together.
        # deltas is an error before activation
        
        deltas_softmax = deltas_kld * sigmoid_grad(pd)
        #deltas_softmax = pd - td

        self.dW_output += np.outer(activation, deltas_softmax)
        self.db_output += deltas_softmax


        deltas_hidden = np.dot(self.W_output, deltas_softmax)

        deltas_hidden *= sigmoid_grad(activation)

        self.dW_1_hidden += np.outer(input_1, deltas_hidden)
        self.dW_2_hidden += np.outer(input_2, deltas_hidden)
        self.db_hidden += deltas_hidden


        deltas_hidden_1 = np.dot(self.W_1_hidden, deltas_hidden) #(n_hidden)
        deltas_hidden_2 = np.dot(self.W_2_hidden, deltas_hidden)
        
        return deltas_hidden_1, deltas_hidden_2


    def predict(self, input_1, input_2):

        lin_output = np.dot(input_1, self.W_1_hidden) + np.dot(input_2, self.W_2_hidden) + self.b_hidden

        activation = sigmoid(lin_output) #(n_hidden,)

        pd= logsoftmax( (np.dot(activation, self.W_logistic) + self.b_logistic).reshape(1, self.numLabels)) #(1, outputDim)

        pd = np.exp(pd.reshape(self.numLabels))

        return pd


    





