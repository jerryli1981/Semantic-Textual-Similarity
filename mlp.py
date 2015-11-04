import theano.tensor as T
import theano
import numpy as np
from utils import *

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        self.inp = input

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
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

            self.W_1 = theano.shared(value=W_1_values, name='W_1', borrow=True)
        else:
            self.W_1 = theano.shared(value=W_1, name='W_1', borrow=True)

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

            self.W_2 = theano.shared(value=W_2_values, name='W_2', borrow=True)
        else:
            self.W_2 = theano.shared(value=W_2, name='W_2', borrow=True)


        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = theano.shared(value=b, name='b', borrow=True)

        self.output = activation(T.dot(input_1, self.W_1) + T.dot(input_2, self.W_2) + self.b)

        self.params = [self.W_1, self.W_2, self.b]


class MLP(object):

    def __init__(self, rng, input_1, input_2, n_in, n_hidden, n_out):


        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input_1=input_1,
            input_2=input_2,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )


        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
        )

        self.L1 = (
            abs(self.hiddenLayer.W_1).sum() + abs(self.hiddenLayer.W_2).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W_1 ** 2).sum() + (self.hiddenLayer.W_2 ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )


        self.kl_divergence = self.logRegressionLayer.kl_divergence

        self.output = self.logRegressionLayer.p_y_given_x

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights

    def __setstate__(self, weights):
        i = iter(weights)
        for p in self.params:
            p.set_value(i.next())
