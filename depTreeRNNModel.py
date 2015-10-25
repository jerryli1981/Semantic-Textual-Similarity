import numpy as np
import random # for random shuffle train data
from multiprocessing import Pool
from numpy import inf

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy


#this is only work for 2 dimension matrix
def softmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)
    return x

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

def roll_params(params):
    L, WR, WV, b, Wsg, Wsm, bsm = params
    return np.concatenate((L.ravel(), WR.ravel(), WV.ravel(), b.ravel(), Wsg.ravel(), Wsm.ravel(), bsm.ravel()))

def unroll_params(arr, hparams):

    relNum, wvecDim, outputDim, numWords, sim_nhidden = hparams

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

    Wsg = arr[ind : ind + sim_nhidden * wvecDim].reshape((sim_nhidden, wvecDim))
    ind += sim_nhidden * wvecDim

    Wsm = arr[ind : ind + outputDim*sim_nhidden].reshape( (outputDim, sim_nhidden))
    ind += outputDim*sim_nhidden

    bsm = arr[ind : ind + outputDim].reshape(outputDim,)

    return (L, WR, WV, b, Wsg, Wsm, bsm)

def unwrap_self_forwardBackwardProp(arg, **kwarg):
    return depTreeRnnModel.forwardBackwardProp(*arg, **kwarg)

class depTreeRnnModel:

    def __init__(self, relNum, wvecDim, outputDim):

        self.relNum = relNum
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.sim_nhidden = 50

    def initialParams(self, word2vecs):

        #generate the same random number
        np.random.seed(12341)

        r = np.sqrt(6)/np.sqrt(201)

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.numWords = word2vecs.shape[1]
        # scale by 0.01 can pass gradient check
        self.L = word2vecs[:self.wvecDim, :]

        # Relation layer parameters
        #self.WR = 0.01*np.random.randn(self.relNum, self.wvecDim, self.wvecDim)
        self.WR = np.random.rand(self.relNum, self.wvecDim, self.wvecDim) * 2 * r -r

        # Hidden layer parameters 
        #self.WV = 0.01*np.random.randn(self.wvecDim, self.wvecDim)
        self.WV = np.random.rand(self.wvecDim, self.wvecDim) * 2 * r - r
        self.b = np.zeros((self.wvecDim))


        #Sigmoid weights
        #self.Wsg = 0.01*np.random.randn(self.sim_nhidden, self.wvecDim)
        self.Wsg = np.random.rand(self.sim_nhidden, self.wvecDim) * 2 * r - r

        # Softmax weights
        #self.Wsm = 0.01*np.random.randn(self.outputDim,self.sim_nhidden) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.Wsm = np.random.rand(self.outputDim, self.sim_nhidden) * 2 * r - r
        self.bsm = np.zeros((self.outputDim))

        self.stack = [self.L, self.WR, self.WV, self.b, self.Wsg, self.Wsm, self.bsm]

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
        self.dWsg = np.zeros((self.sim_nhidden, self.wvecDim))
        self.dWsm = np.zeros((self.outputDim, self.sim_nhidden))
        self.dbsm = np.zeros(self.outputDim)


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
        td = np.zeros(self.outputDim+1) 
        sim = tree.score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            td[floor] = 1
        else:
            td[floor] = ceil-sim
            td[ceil] = sim-floor

        td = td[1:]
        td += 1e-8
        #compute similarity
        tree.root.sigAct = sigmoid(np.dot(self.Wsg, tree.root.hAct)) #(sim_nhidden,)
        pd= logsoftmax((np.dot(self.Wsm, tree.root.sigAct) + self.bsm).reshape(1, self.outputDim)) #(1, outputDim)
        #KL divergence loss, loss(x, target) = \sum(target_i * (log(target_i) - x_i))
        log_td = np.log(td)
        log_td[log_td == -inf] = 0
        cost = np.dot(td, log_td-pd.reshape(self.outputDim)) / self.outputDim

        tree.root.pd = np.exp(pd.reshape(self.outputDim))
        tree.root.td = td

        if test:
            predictScore = np.exp(pd.reshape(self.outputDim)).dot(np.array([1,2,3,4,5]))
            return cost, float("{0:.2f}".format(predictScore)), tree.score
        else:
            return cost

    def backProp(self, tree):

        to_do = []
        to_do.append(tree.root)

        norm = -1.0/self.outputDim
        sim_grad = norm * tree.root.td
        sim_grad[sim_grad == -0.] = 0
        
        #softmax gradient
        deltas_sm = sim_grad * (tree.root.pd * (1-tree.root.pd))

        self.dWsm += np.outer(deltas_sm,tree.root.sigAct)
        self.dbsm += deltas_sm

        deltas_sg = np.dot(self.Wsm.T, deltas_sm) #(n_hidden)
  
        
        #f = f*(1-f)   

        deltas_sg *= tree.root.sigAct*(1-tree.root.sigAct)

        self.dWsg += np.outer(deltas_sg, tree.root.hAct)

        tree.root.deltas = np.dot(self.Wsg.T,deltas_sg)

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

        return cost, roll_params((self.dL, self.dWR, self.dWV, self.db, self.dWsg, self.dWsm, self.dbsm))

    def costAndGrad(self, mbdata, rho=1e-4): 


        cost, grad = self.forwardBackwardProp(mbdata)

        hparams = (self.relNum, self.wvecDim, self.outputDim, self.numWords, self.sim_nhidden)

        self.dL, self.dWR, self.dWV, self.db, self.dWsg, self.dWsm, self.dbsm = unroll_params(grad, hparams)

        # scale cost and grad by mb size
        scale = (1./len(mbdata))
        for v in range(self.numWords):
            self.dL[:,v] *= scale
            
        # Add L2 Regularization 
        cost += (rho/2)*np.sum(self.WV**2)
        cost += (rho/2)*np.sum(self.WR**2)
        cost += (rho/2)*np.sum(self.Wsg**2)
        cost += (rho/2)*np.sum(self.Wsm**2)

        return scale*cost, [self.dL,scale*(self.dWR + rho*self.WR),
                                   scale*(self.dWV + rho*self.WV),
                                   scale*self.db,
                                   scale*(self.dWsg+rho*self.Wsg),
                                   scale*(self.dWsm+rho*self.Wsm),
                                   scale*self.dbsm]
            
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

        stack = [self.L, self.WR, self.WV, self.b, self.Wsg, self.Wsm, self.bsm]

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
        #print "Pearson correlation %f"%(pearsonr(corrects,guesses)[0])

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

if __name__ == '__main__':

    import dependency_tree as treeM  

    
    train = treeM.loadTrees()
    print "train number %d"%len(train)
     
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

