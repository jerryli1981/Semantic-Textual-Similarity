import numpy as np
import random # for random shuffle train data
from multiprocessing import Pool

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy


#this is only work for 2 dimension matrix
def softmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)

    return x

def roll_params(params):
    L, WR, WV, b, Ws, bs = params
    return np.concatenate((L.ravel(), WR.ravel(), WV.ravel(), b.ravel(), Ws.ravel(), bs.ravel()))

def unroll_params(arr, hparams):

    relNum, wvecDim, outputDim, numWords = hparams

    ind = 0

    d = wvecDim*wvecDim

    L = arr[ind : ind + numWords*wvecDim].reshape( (wvecDim, numWords) )
    ind +=numWords*wvecDim

    WR = arr[ind : ind + relNum*d].reshape( (relNum, wvecDim, wvecDim) )
    ind += relNum*d

    WV = arr[ind : ind + d].reshape( (wvecDim, wvecDim) )
    ind += d

    b = arr[ind : ind + wvecDim].reshape(wvecDim,)
    ind += wvecDim

    Ws = arr[ind : ind + outputDim*wvecDim].reshape( (outputDim, wvecDim))
    ind += outputDim*wvecDim

    bs = arr[ind : ind + outputDim].reshape(outputDim,)

    return (L, WR, WV, b, Ws, bs)

def unwrap_self_forwardBackwardProp(arg, **kwarg):
    return depTreeRnnModel.forwardBackwardProp(*arg, **kwarg)

class depTreeRnnModel:

    def __init__(self, relNum, wvecDim, outputDim):

        self.relNum = relNum
        self.wvecDim = wvecDim
        self.outputDim = outputDim

    def initialParams(self, word2vecs):

        #generate the same random number
        np.random.seed(12341)

        r = np.sqrt(6)/np.sqrt(201)

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.numWords = word2vecs.shape[1]
        # scale by 0.01 can pass gradient check
        self.L = 0.01*word2vecs[:self.wvecDim, :]

        # Hidden layer parameters 
        #self.WV = 0.01*np.random.randn(self.wvecDim, self.wvecDim)
        self.WV = 0.01*np.random.rand(self.wvecDim, self.wvecDim) * 2 * r - r

        # Relation layer parameters
        #self.WR = 0.01*np.random.randn(self.relNum, self.wvecDim, self.wvecDim)
        self.WR = 0.01*np.random.rand(self.relNum, self.wvecDim, self.wvecDim) * 2 * r -r

        self.b = np.zeros((self.wvecDim))


        # Softmax weights
        #self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.Ws = 0.01*np.random.rand(self.outputDim, self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.WR, self.WV, self.b, self.Ws, self.bs]

    def initialGrads(self):

        #create with default_factory : defaultVec = lambda : np.zeros((wvecDim,))
        #make the defaultdict useful for building a dictionary of np array
        #this is very good to save memory, However, this may not support for multiprocess due to pickle error

        #defaultVec = lambda : np.zeros((wvecDim,))
        #dL = collections.defaultdict(defaultVec)
        self.dL = np.zeros((self.wvecDim, self.numWords))

        self.dWR = np.zeros((self.relNum, self.wvecDim, self.wvecDim))
        self.dWV = np.zeros((self.wvecDim, self.wvecDim))
        self.db = np.zeros(self.wvecDim)
        self.dWs = np.zeros((self.outputDim,self.wvecDim))
        self.dbs = np.zeros(self.outputDim)


    def forwardProp(self, tree, test=False):

        #because many training epoch. 
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
                curr.hAct= np.tanh(np.dot(self.WV,curr.vec) + self.b)
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
                        pass
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

        """

        sim = 0.25 * (tree.score-1) * 4 + 1
        ceil = np.ceil(sim)
        floor = np.floor(sim)

        target_distribution = np.zeros((1, self.outputDim))

        for i in range(self.outputDim):
            if i == ceil and ceil == floor:
                target_distribution[0, i] = 1
            elif i == floor:
                target_distribution[0,i] = ceil - sim
            elif i == ceil:
                target_distribution[0,i] = sim - floor
        """


        predicted_distribution= softmax((np.dot(self.Ws, tree.root.hAct) + self.bs).reshape(1, self.outputDim))
        tree.root.pd = predicted_distribution.reshape(self.outputDim)
        tree.root.td = target_distribution.reshape(self.outputDim)

        #cost = -np.dot(target_distribution, np.log(predicted_distribution).T)
        cost = entropy(target_distribution.reshape(outputDim,),predicted_distribution.reshape(outputDim,))

        #assert cost1[0,0] == cost2, "they should equal"
        
        #correctLabel = np.argmax(target_distribution)
        #guessLabel = np.argmax(predicted_distribution)

        predictScore = predicted_distribution.reshape(self.outputDim,).dot(np.array([1,2,3,4,5]))
        #return cost, predictScore, tree.score

        if test:
            return cost, float("{0:.2f}".format(predictScore)), tree.score
        else:
            return cost

    def backProp(self, tree):

        to_do = []
        to_do.append(tree.root)

        deltas = tree.root.pd-tree.root.td
        self.dWs += np.outer(deltas,tree.root.hAct)
        self.dbs += deltas

        tree.root.deltas = np.dot(self.Ws.T,deltas)

        while to_do:

            curr = to_do.pop(0)

            if len(curr.kids) == 0:
                self.dL[:, curr.index] += curr.deltas
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


    def forwardBackwardProp(self, mbdata):
        cost = 0.0
        for tree in mbdata: 
            cost += self.forwardProp(tree)
            self.backProp(tree)

        return cost, roll_params((self.dL, self.dWR, self.dWV, self.db, self.dWs, self.dbs))

    def costAndGrad(self, mbdata, rho=1e-4): 


        cost, grad = self.forwardBackwardProp(mbdata)

        hparams = (self.relNum, self.wvecDim, self.outputDim, self.numWords)

        self.dL, self.dWR, self.dWV, self.db, self.dWs, self.dbs = unroll_params(grad, hparams)

        # scale cost and grad by mb size
        scale = (1./len(mbdata))
        for v in range(self.numWords):
            self.dL[:,v] *= scale
            
        # Add L2 Regularization 
        cost += (rho/2)*np.sum(self.WV**2)
        cost += (rho/2)*np.sum(self.WR**2)
        cost += (rho/2)*np.sum(self.Ws**2)

        return scale*cost, [self.dL,scale*(self.dWR + rho*self.WR),
                                   scale*(self.dWV + rho*self.WV),
                                   scale*self.db,
                                   scale*(self.dWs+rho*self.Ws),scale*self.dbs]
            
    def costAndGrad_MultiP(self, batchData, numProc, rho=1e-4):

        miniBatchSize = len(batchData) / numProc

        pool = Pool(processes = numProc)
        

        miniBatchData = [batchData[i:i+miniBatchSize] for i in range(0, len(batchData), miniBatchSize)]

        result = pool.map(unwrap_self_forwardBackwardProp, zip([self]*len(miniBatchData), miniBatchData))

        pool.close() #no more processed accepted by this pool
        pool.join() #wait until all processes are finished

        cost = 0.
        grad = None
        for mini_cost, mini_grads in result:
            cost += mini_cost
            if grad is None:
                grad = mini_grads
            else:
                grad += mini_grads

        hparams = (self.relNum, self.wvecDim, self.outputDim, self.numWords)

        self.dL, self.dWR, self.dWV, self.db, self.dWs, self.dbs = unroll_params(grad, hparams)

        # scale cost and grad by mb size
        scale = (1./len(batchData))
        for v in range(self.numWords):
            self.dL[:,v] *= scale
            
        # Add L2 Regularization 
        cost += (rho/2)*np.sum(self.WV**2)
        cost += (rho/2)*np.sum(self.WR**2)
        cost += (rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dWR + rho*self.WR),
                                   scale*(self.dWV + rho*self.WV),
                                   scale*self.db,
                                   scale*(self.dWs+rho*self.Ws),scale*self.dbs]

    #Minbatch stochastic gradient descent
    def train(self, trainData, alpha, batchSize, numProcess=None):

        random.shuffle(trainData)

        batches = [trainData[x : x + batchSize] for x in xrange(0, len(trainData), batchSize)]

        stack = [self.L, self.WR, self.WV, self.b, self.Ws, self.bs]

        #print "Using adagrad..."
        epsilon = 1e-8

        gradt = [epsilon + np.zeros(W.shape) for W in stack]

        self.initialGrads()

        for batchData in batches:

            if numProcess != None:
                cost, grad = self.costAndGrad_MultiP(batchData, numProcess)
            else:
                cost, grad = self.costAndGrad(batchData)

            # trace = trace+grad.^2
            gradt[1:] = [gt+g**2 for gt,g in zip(gradt[1:],grad[1:])]

            # update = grad.*trace.^(-1/2)
            update =  [g*(1./np.sqrt(gt)) for gt,g in zip(gradt[1:],grad[1:])]

            # handle dictionary separately
            dL = grad[0]
            dLt = gradt[0]
            for j in range(self.numWords):
                dLt[:,j] = dLt[:,j] + dL[:,j]**2
                dL[:,j] = dL[:,j] * (1./np.sqrt(dLt[:,j]))

            scale = -alpha

            update = [dL] + update


            self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

            for j in range(self.numWords):
                self.L[:,j] += scale*dL[:,j]


        for tree in trainData:
                tree.resetFinished()


    def predict(self, trees):

        cost = 0
        corrects = []
        guesses = []
        for tree in trees:
            c, guess, correct= self.forwardProp(tree, test=True)
            cost += c
            corrects.append(correct)
            guesses.append(guess)

        print corrects[:10]
        print guesses[:10]
        print "Cost %f"%(cost/len(trees))    
        print "Pearson correlation %f"%(pearsonr(corrects,guesses)[0])

        for tree in trees:
                tree.resetFinished()

        #print "Spearman correlation %f"%(spearmanr(corrects,guesses)[0])
        return pearsonr(corrects,guesses)[0]

    def check_grad(self,data,epsilon=1e-6):

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
    word2vecs = treeM.loadWord2VecMap()

    wvecDim = 10
    outputDim = 5

    rnn = depTreeRnnModel(relNum, wvecDim, outputDim)
    rnn.initialParams(word2vecs)
    
    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)

