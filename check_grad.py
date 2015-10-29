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
    
    print "Numerical gradient check..."
    mbData = trainTrees[:4]
    optimizer = SGD(rep_model=rnn, rng=rng, alpha=0.01, optimizer='adadelta')
    check_grad(optimizer, mbData)
