#these are for mlp_myself
# LogSoftmax is defined as f_i(x) = log(1/a exp(x_i)), where a = sum_j exp(x_j).
import numpy as np

def logsoftmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)
    return np.log(x)

def sigmoid(x):
    """ Sigmoid function """
    x = 1/(1+np.exp(-x))    
    return x

def derivative_sigmoid(f):
    """ Sigmoid gradient function """
    f = f*(1-f)
    return f

def derivative_softmax(f):
    """ Softmax gradient function """
    f = f*(1-f)
    return f

def derivative_tanh(f):
	return 1 - f**2