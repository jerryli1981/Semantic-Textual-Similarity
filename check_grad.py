import numpy as np

def check_wordEmbedding_grad(optimizer,data,epsilon=1e-6):
    raise "current not work"
    err1 = 0.0
    count = 0.0
    
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

def check_param_grad(optimizer, data, epsilon=1e-6):


    cost, mlp_grad = optimizer.costAndGrad_theano_single_grad(data)
    
    err1 = 0.0
    count = 0.0

    hw_1 = optimizer.classifier.hiddenLayer.params[0].eval()
    hw_2 = optimizer.classifier.hiddenLayer.params[1].eval()
    hb = optimizer.classifier.hiddenLayer.params[2].eval()
    log_w = optimizer.classifier.logRegressionLayer.params[0].eval()
    log_b = optimizer.classifier.logRegressionLayer.params[1].eval()

    mlp_stack = [hw_1, hw_2, hb, log_w, log_b]

    grad = optimizer.rep_model.dstack + mlp_grad

    stack = optimizer.rep_model.stack + mlp_stack

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
                    costP,_ = optimizer.costAndGrad_theano_single_grad(data)
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
    
    print "Numerical gradient check..."
    import dependency_tree as tr     
    trainTrees = tr.loadTrees("train")
    print "train number %d"%len(trainTrees)
    
    mbData = trainTrees[:4]

    from optimization import Optimization
     
    optimizer = Optimization(alpha=0.01, optimizer="sgd")

    wvecDim = 10
    outputDim = 5
    hiddenDim = 50

    optimizer.initial_RepModel(tr, "RNN", wvecDim)

    optimizer.initial_theano_mlp(hiddenDim, outputDim, batchMLP=False)

    check_param_grad(optimizer, mbData)
