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

# derivative of normalized tanh
def derivative_norm_tanh(x):
    norm = np.linalg.norm(x)
    y = x - np.power(x, 3)
    dia = np.diag((1 - np.square(x)).flatten()) / norm
    pro = y.dot(x.T) / np.power(norm, 3)
    out = dia - pro
    return out

def norm_tanh(x):
    x = np.tanh(x)
    x /= np.linalg.norm(x)

    return x

def derivative_tanh(f):
	return 1 - f**2