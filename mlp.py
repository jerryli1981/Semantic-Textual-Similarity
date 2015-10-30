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

    def kl_divergence_single(self, y):
        newshape=(T.shape(self.output)[1],)
        x = T.reshape(self.output, newshape)
        return T.dot(y, (T.log(y) - T.log(x)).T) / self.numLabels

    def kl_divergence_batch(self, y):
        return T.sum(y * (T.log(y) - T.log(self.output))) / (self.numLabels * y.shape[0])


    def predict_p(self, x_1, x_2):

        W_1_hidden, W_2_hidden, b_hidden = self.hiddenLayer.params

        W_lg, b_lg = self.logRegressionLayer.params

        output = T.nnet.sigmoid( T.dot(x_1, W_1_hidden) + T.dot(x_2, W_2_hidden) + b_hidden)

        p_y_given_x = T.nnet.softmax(T.dot(output, W_lg) + b_lg)

        return p_y_given_x
   

class my_mlp(object):

    def __init__(self, rng, n_in, n_hidden, n_out):

        self.numLabels = n_out

        self.W_1_hidden = rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_hidden)),
                    high=np.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)
                )

        self.W_2_hidden = rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_hidden)),
                    high=np.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)
                )

        self.b_hidden = np.zeros((n_hidden,))

        self.W_logistic = rng.uniform(
                    low=-np.sqrt(6. / (n_hidden + n_out)),
                    high=np.sqrt(6. / (n_hidden + n_out)),
                    size=(n_hidden, n_out)
                )

        self.b_logistic = np.zeros((n_out,))



        self.dW_1_hidden = np.zeros((n_in, n_hidden))

        self.dW_2_hidden = np.zeros((n_in, n_hidden))

        self.db_hidden = np.zeros((n_hidden,))

        self.dW_logistic = np.zeros((n_hidden, n_out))

        self.db_logistic = np.zeros((n_out,))

        self.params = [self.W_1_hidden, self.W_2_hidden, self.b_hidden, self.W_logistic, self.b_logistic]

        self.dstack = [self.dW_1_hidden, self.dW_2_hidden, self.db_hidden, self.dW_logistic, self.db_logistic]
        

    def forwardProp(self, input_1, input_2, td):

        lin_output = np.dot(input_1, self.W_1_hidden) + np.dot(input_2, self.W_2_hidden) + self.b_hidden

        activation = sigmoid(lin_output) #(n_hidden,)

        pd= logsoftmax( (np.dot(activation, self.W_logistic) + self.b_logistic).reshape(1, self.numLabels)) #(1, outputDim)
        #KL divergence loss, loss(x, target) = \sum(target_i * (log(target_i) - x_i))
        log_td = np.log(td)

        cost = np.dot(td, log_td - pd.reshape(self.numLabels) ) / self.numLabels

        pd = np.exp(pd.reshape(self.numLabels))

        return cost, activation, pd

    def backwardProp(self, input_1, input_2, activation, pd, td):

        norm = -1.0/self.numLabels
        deltas_kld = norm * td
        deltas_kld[deltas_kld == -0.] = 0
        
        #softmax gradient
        deltas_logistic = deltas_kld * (pd * (1 - pd))

        self.dW_logistic += np.outer(activation, deltas_logistic)
        self.db_logistic += deltas_logistic


        deltas_hidden = np.dot(self.W_logistic, deltas_logistic)

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


