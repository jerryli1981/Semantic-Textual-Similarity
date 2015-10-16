import numpy as np
import collections

def softmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)

    return x

class RNN:

    def __init__(self, relNum, wvecDim,outputDim,numWords,mbSize=30,rho=1e-4):
        self.relNum = relNum
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self, word2vecs):
        np.random.seed(12341)

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        word2vecs = word2vecs[:self.wvecDim, :]

        self.L = 0.01*word2vecs

        # Hidden layer parameters 
        self.WV = 0.01*np.random.randn(self.wvecDim, self.wvecDim)

        # Relation layer parameters
        self.WR = 0.01*np.random.randn(self.relNum, self.wvecDim, self.wvecDim)

        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))


        #here self.WR should ahead of self.WV, unless gradient check will fail on 
        #non-broadcastable output operand with shape (1) doesn't match the broadcast shape (1,1)
        self.stack = [self.L, self.WR, self.WV, self.b, self.Ws, self.bs]

        # Gradients
        self.dWV = np.empty(self.WV.shape)
        self.dWR = np.empty(self.WR.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self, mbdata, test=False): 

        cost = 0.0

        self.L, self.WR, self.WV, self.b, self.Ws, self.bs = self.stack

        # Zero gradients
        self.dWV[:] = 0
        self.dWR[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0

        #create with default_factory : self.defaultVec = lambda : np.zeros((wvecDim,))
        self.dL = collections.defaultdict(self.defaultVec)


        # Forward prop each tree in minibatch
        corrects = []
        guesses = []
        for tree in mbdata: 
            c, correct, guess = self.forwardProp(tree)
            corrects.append(correct)
            guesses.append(guess)
            cost += c

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.WV**2)
        cost += (self.rho/2)*np.sum(self.WR**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        # the grad order should consistent with stack order
        if test:
            return scale*cost,[self.dL,scale*(self.dWR + self.rho*self.WR),
                                   scale*(self.dWV + self.rho*self.WV),
                                   scale*self.db,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs],corrects,guesses
        else:
            return scale*cost,[self.dL,scale*(self.dWR + self.rho*self.WR),
                                   scale*(self.dWV + self.rho*self.WV),
                                   scale*self.db,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]


    def forwardProp(self,tree):
        #because many training iterations. 
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
                curr.hAct= np.tanh(self.WV.dot(curr.vec) + self.b)
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
                        rel_vec = self.WR[rel.index]
                        sum += rel_vec.dot(tree.nodes[i].hAct) 

                    curr.hAct = np.tanh(sum + self.WV.dot(curr.vec) + self.b)
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        # compute target distribution   
        if tree.score == 5.0:
            tree.score = 4.99999 
        target_distribution = np.zeros((1, self.outputDim))
        for i in xrange(self.outputDim):
            score_floor = np.floor(tree.score)
            if i == score_floor + 1:
                target_distribution[0,i] = tree.score - score_floor
            elif i == score_floor:
                target_distribution[0,i] = score_floor - tree.score + 1
            else:
                target_distribution[0,i] = 0  

        predicted_distribution= softmax((np.dot(self.Ws, tree.root.hAct) + self.bs).reshape(1, self.outputDim))
        tree.root.pd = predicted_distribution.reshape(self.outputDim)
        tree.root.td = target_distribution.reshape(self.outputDim)
        cost = -np.dot(target_distribution, np.log(predicted_distribution).T)
        
        correctLabel = np.argmax(target_distribution)
        guessLabel = np.argmax(predicted_distribution)
        predictScore = predicted_distribution.reshape(self.outputDim,).dot(np.array([1,2,3,4,5]))
        return cost[0,0], predictScore, tree.score

    def backProp(self,tree):

        to_do = []
        to_do.append(tree.root)

        deltas = tree.root.pd-tree.root.td
        self.dWs += np.outer(deltas,tree.root.hAct)
        self.dbs += deltas

        tree.root.deltas = np.dot(self.Ws.T,deltas)

        while to_do:

            curr = to_do.pop(0)

            if len(curr.kids) == 0:

                self.dL[curr.index] += curr.deltas

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

        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

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

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    __DEBUG__ = False
    if __DEBUG__:
        import pdb
        pdb.set_trace()

    import dependency_tree as treeM    
    
    train = treeM.loadTrees()
     
    numW = len(treeM.loadWordMap())

    relMap = treeM.loadRelMap()
    relNum = len(relMap)

    wvecDim = 10
    outputDim = 5

    
    #mb : mini batch
    rnn = RNN(relNum,wvecDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)
