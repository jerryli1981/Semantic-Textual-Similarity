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

def check_grad_backup(self,data,epsilon=1e-6):

    self.initialGrads()
    cost, grad = self.costAndGrad(data)
    err1 = 0.0
    count = 0.0

    print "Checking dW... (might take a while)"
    for W,dW in zip(self.stack[1:],grad[1:]):
        W = W[...,None,None] # add dimension since bias is flat
        dW = dW[...,None,None] 
        for i in xrange(W.shape[0]):
            for j in xrange(W.shape[1]):
                for k in xrange(W.shape[2]):
                    W[i,j,k] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j,k] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j,k] - numGrad)
                    #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                    err1+=err
                    count+=1

    if 0.001 > err1/count:
        print "Grad Check Passed for dW"
    else:
        print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)
    
    """
    # check dL separately since dict
    dL = grad[0]
    L = self.stack[0]
    err2 = 0.0
    count = 0.0
    print "Checking dL..."
    for j in range(self.numWords):
        for i in xrange(L.shape[0]):
            L[i,j] += epsilon
            costP,_ = self.costAndGrad(data)
            L[i,j] -= epsilon
            numGrad = (costP - cost)/epsilon
            err = np.abs(dL[i,j] - numGrad)
            #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
            err2+=err
            count+=1

    if 0.001 > err2/count:
        print "Grad Check Passed for dL"
    else:
        print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)
    """

def check_grad(sgd, data, epsilon=1e-6):
        cost, grad = sgd.costAndGrad(data)
        err1 = 0.0
        count = 0.0

        hw_1 = sgd.classifier.hiddenLayer.params[0].eval()
        hw_2 = sgd.classifier.hiddenLayer.params[1].eval()
        hb = sgd.classifier.hiddenLayer.params[2].eval()
        log_w = sgd.classifier.logRegressionLayer.params[0].eval()
        log_b = sgd.classifier.logRegressionLayer.params[1].eval()

        mlp_stack = [hw_1, hw_2, hb, log_w, log_b]

        stack = sgd.rep_model.stack + mlp_stack

        print "Checking dW... (might take a while)"
        idx =0
        for W,dW in zip(stack[1:],grad[1:]):
            print idx
            idx += 1
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = sgd.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1

        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=50)
    parser.add_argument("--step",dest="step",type=float,default=1e-2)
    parser.add_argument("--outputDim",dest="outputDim",type=int,default=5)
    parser.add_argument("--wvecDim",dest="wvecDim",type=int,default=30)
    parser.add_argument("--outFile",dest="outFile",type=str, default="models/test.bin")
    parser.add_argument("--numProcess",dest="numProcess",type=int,default=None)

 
    args = parser.parse_args()

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

    rnn = depTreeRnnModel(relNum, wvecDim, outputDim)
    #rnn = depTreeLSTMModel(wvecDim, outputDim)

    rnn.initialParams(word2vecs, rng=rng)

    """
    print "Numerical gradient check..."
    mbData = trainTrees[:4]
    optimizer = SGD(rep_model=rnn, rng=rng, alpha=0.01, optimizer='adadelta')
    check_grad(optimizer, mbData)
    """

    
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

    



