import optparse
import cPickle as pickle

import dependency_tree as tr
import time
import numpy as np

import collections
import random # for random shuffle train data
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy



from multiprocessing import Pool


def initParams(word2vecs, hparams):

    relNum, wvecDim, outputDim, numWords = hparams

    #generate the same random number
    np.random.seed(12341)

    r = np.sqrt(6)/np.sqrt(201)

    # Word vectors
    #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
    L = word2vecs[:wvecDim, :]

    # Hidden layer parameters 
    #self.WV = 0.01*np.random.randn(self.wvecDim, self.wvecDim)
    WV = np.random.rand(wvecDim, wvecDim) * 2 * r - r

    # Relation layer parameters
    #self.WR = 0.01*np.random.randn(self.relNum, self.wvecDim, self.wvecDim)
    WR = np.random.rand(relNum, wvecDim, wvecDim) * 2 * r -r

    b = np.zeros((wvecDim))


    # Softmax weights
    #self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
    Ws = np.random.rand(outputDim, wvecDim)
    bs = np.zeros((outputDim))


    #here self.WR should ahead of self.WV, unless gradient check will fail on 
    #non-broadcastable output operand with shape (1) doesn't match the broadcast shape (1,1)
    return (L, WR, WV, b, Ws, bs)

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


def init_grads(hparams):

    relNum, wvecDim, outputDim, numWords = hparams

    #create with default_factory : defaultVec = lambda : np.zeros((wvecDim,))
    #make the defaultdict useful for building a dictionary of np array
    #this is very good to save memory, However, this may not support for multiprocess due to pickle error

    #defaultVec = lambda : np.zeros((wvecDim,))
    #dL = collections.defaultdict(defaultVec)
    dL = np.zeros((wvecDim, numWords))
    dWR = np.zeros((relNum, wvecDim, wvecDim))
    dWV = np.zeros((wvecDim, wvecDim))
    db = np.zeros(wvecDim)
    dWs = np.zeros((outputDim,wvecDim))
    dbs = np.zeros(outputDim)

    return (dL, dWR, dWV, db, dWs, dbs)


#this is only work for 2 dimension matrix
def softmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)

    return x

def forwardProp(hparams, params, tree, test=False):

        #because many training epoch. 
        tree.resetFinished()

        L, WR, WV, b, Ws, bs = params

        relNum, wvecDim, outputDim, numWords = hparams

        cost  =  0.0 

        to_do = []
        to_do.append(tree.root)

        while to_do:
        
            curr = to_do.pop(0)
            curr.vec = L[:, curr.index]

            # node is leaf
            if len(curr.kids) == 0:
                # activation function is the normalized tanh
                curr.hAct= np.tanh(np.dot(WV,curr.vec) + b)
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
                    sum = np.zeros((wvecDim))
                    for i, rel in curr.kids:
                        pass
                        rel_vec = WR[rel.index]
                        sum += rel_vec.dot(tree.nodes[i].hAct) 

                    curr.hAct = np.tanh(sum + WV.dot(curr.vec) + b)
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        # compute target distribution   
        """
        if tree.score == 5.0:
            tree.score = 4.99999
        """ 

        tree.score = 0.25 * (tree.score-1)

        sim = tree.score * 4 + 1
        ceil = np.ceil(sim)
        floor = np.floor(sim)

        target_distribution = np.zeros((1, outputDim))

        """
        for i in range(outputDim):
            score_floor = np.floor(tree.score)
            if i == score_floor + 1:
                target_distribution[0,i] = tree.score - score_floor
            elif i == score_floor:
                target_distribution[0,i] = score_floor - tree.score + 1
            else:
                target_distribution[0,i] = 0  
        """


        for i in range(outputDim):
            if i == ceil and ceil == floor:
                target_distribution[0, i] = 1
            elif i == floor:
                target_distribution[0,i] = ceil - sim
            elif i == ceil:
                target_distribution[0,i] = sim - floor

        predicted_distribution= softmax((np.dot(Ws, tree.root.hAct) + bs).reshape(1, outputDim))
        tree.root.pd = predicted_distribution.reshape(outputDim)
        tree.root.td = target_distribution.reshape(outputDim)

        cost = -np.dot(target_distribution, np.log(predicted_distribution).T)
        #cost = entropy(target_distribution.reshape(outputDim,),predicted_distribution.reshape(outputDim,))

        #assert cost1[0,0] == cost2, "they should equal"
        
        #correctLabel = np.argmax(target_distribution)
        #guessLabel = np.argmax(predicted_distribution)

        predictScore = predicted_distribution.reshape(outputDim,).dot(np.array([1,2,3,4,5]))
        #return cost, predictScore, tree.score

        if test:
            return cost, predictScore, tree.score
        else:
            return cost

def backProp(grads, params, tree):

    dL, dWR, dWV, db, dWs, dbs= grads
    L, WR, WV, b, Ws, bs = params

    to_do = []
    to_do.append(tree.root)

    deltas = tree.root.pd-tree.root.td
    dWs += np.outer(deltas,tree.root.hAct)
    dbs += deltas

    tree.root.deltas = np.dot(Ws.T,deltas)

    while to_do:

        curr = to_do.pop(0)

        if len(curr.kids) == 0:
            dL[:, curr.index] += curr.deltas
        else:

            # derivative of tanh
            curr.deltas *= (1-curr.hAct**2)

            dWV += np.outer(curr.deltas, curr.vec)
            db += curr.deltas

            for i, rel in curr.kids:

                kid = tree.nodes[i]
                to_do.append(kid)

                dWR[rel.index] += np.outer(curr.deltas, kid.hAct)

                rel_vec = WR[rel.index]
                kid.deltas = np.dot(rel_vec.T, curr.deltas)


def costAndGrad(data): 

    hparams, params = data[0]
    mbdata = data[1]

    grads = init_grads(hparams)

    params = unroll_params(params, hparams)

    cost = 0.0

    for tree in mbdata: 
        cost += forwardProp(hparams, params, tree)
        backProp(grads, params, tree)
    
    r_grads = roll_params(grads)
        
    return cost, r_grads


def costAndGrad_MultiP(numProc, batchData, hparams, params, miniBatchSize, rho=1e-4):
    pool = Pool(processes = numProc)

    oparams = (hparams, params)

    miniBatchData = [batchData[i:i+miniBatchSize] for i in range(0, len(batchData), miniBatchSize)]

    to_map = []

    for item in miniBatchData:
        to_map.append((oparams, item))

    result = pool.map(costAndGrad,to_map)

    pool.close() #no more processed accepted by this pool
    pool.join() #wait until all processes are finished

    cost = 0.
    grad = None

    for mini_cost, mini_r_grads in result:
        cost += mini_cost
        if grad is None:
            grad = mini_r_grads
        else:
            grad += mini_r_grads

    L, WR, WV, b, Ws, bs = unroll_params(params, hparams)
    
    # Add L2 Regularization 
    reg_cost = 0.
    reg_cost += (rho/2)*np.sum(WV**2)
    reg_cost += (rho/2)*np.sum(WR**2)
    reg_cost += (rho/2)*np.sum(Ws**2)

    # scale cost and grad by mb size

    dL, dWR, dWV, db, dWs, dbs = unroll_params(grad,hparams)


    """
    scale = (1./len(batchData))
    for v in range(hparams[3]):
        dL[:,v] *= scale

    cost = scale*cost + reg_cost

    dWR = scale*(dWR + rho*WR)
    dWV = scale*(dWV + rho*WV)
    db = scale*db
    dWs = scale*(dWs+rho*Ws)
    dbs = scale*dbs
    """
    
    #r_grads = roll_params((dL, dWR, dWV, db, dWs, dbs))

    return cost, (dL, dWR, dWV, db, dWs, dbs)

#Minbatch stochastic gradient descent
def sgd(trainData, alpha, batchSize, numProc, hparams, r_params):

    random.shuffle(trainData)

    batches = [trainData[x : x + batchSize] for x in xrange(0, len(trainData), batchSize)]

    miniBatchSize = batchSize / numProc

    stack = unroll_params(r_params, hparams)

    #print "Using adagrad..."
    epsilon = 1e-8

    gradt = [epsilon + np.zeros(W.shape) for W in stack]

    for batchData in batches:

        cost, grad = costAndGrad_MultiP(numProc, batchData, hparams, r_params, miniBatchSize)

        # trace = trace+grad.^2
        gradt[1:] = [gt+g**2 for gt,g in zip(gradt[1:],grad[1:])]
        # update = grad.*trace.^(-1/2)
        update =  [g*(1./np.sqrt(gt)) for gt,g in zip(gradt[1:],grad[1:])]
        # handle dictionary separately
        dL = grad[0]
        dLt = gradt[0]
        for j in range(hparams[3]):
            dLt[:,j] = dLt[:,j] + dL[:,j]**2
            dL[:,j] = dL[:,j] * (1./np.sqrt(dLt[:,j]))

        update = [dL] + update

        
        scale = -alpha

        scale = 0.1
        params = list(unroll_params(r_params, hparams))
        params[1:] = [P+scale*dP for P,dP in zip(params[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in range(hparams[3]):
            params[0][:,j] += scale*dL[:,j]

        r_params = roll_params(params)

            
def train(args = None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")

    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)

    parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)

    
    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="RNN")

    parser.add_option("--crossValidation",dest="crossValidation",type="int",default=10)

    parser.add_option("--numProcess",dest="numProcess",type="int",default=4)

    (opts, args)=parser.parse_args(args)

    evaluate_accuracy_while_training = True

    opts.numWords = len(tr.loadWordMap())

    opts.relNum = len(tr.loadRelMap())
    
    word2vecs = tr.loadWord2VecMap()

    trainTrees = tr.loadTrees("train")
    devTrees = tr.loadTrees("dev")

    print "train size %s"%len(trainTrees)
    print "dev size %s"%len(devTrees)

    hparams = (opts.relNum, opts.wvecDim, opts.outputDim, opts.numWords)
    r_params = roll_params(initParams(word2vecs, hparams))

    
    for e in range(opts.epochs):
        print "Running epoch %d"%e
        start = time.time()
        sgd(trainTrees, opts.step, opts.minibatch, opts.numProcess, hparams, r_params)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        """
        stack = unroll_params(r_params,hparams)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(stack,fid)
        """

        if evaluate_accuracy_while_training:

            #print "testing on training set real quick"
            #train_accuracies.append(test(opts.outFile,"train",word2vecs,opts.model,trainTrees))
            #print "testing on dev set real quick"
            #dev_accuracies.append(test(opts.outFile,"dev",opts.model,devTrees))
            evl = test(devTrees, hparams, r_params)
            #dev_pearsons.append(evl[0])
            #dev_spearmans.append(evl[1])

            # because tesing need to forward propogation, so clear the fprop flags in trees and dev_trees
            for tree in trainTrees:
                tree.resetFinished()
            for tree in devTrees:
                tree.resetFinished()
            #print "fprop in trees cleared"

def test(trees, hparams, r_params):

    params = unroll_params(r_params,hparams)
        
    opts.relNum, opts.wvecDim, opts.outputDim, opts.numWords = hparams
    cost = 0
    corrects = []
    guesses = []
    for tree in trees:
        c, guess, correct= forwardProp(hparams, params, tree, test=True)
        cost += c
        corrects.append(correct)
        guesses.append(guess)

    print "Cost %f"%(cost/len(trees))    
    print "Pearson correlation %f"%(pearsonr(corrects,guesses)[0])

    #print "Spearman correlation %f"%(spearmanr(corrects,guesses)[0])
    #return pearsonr(corrects,guesses)[0],spearmanr(corrects,guesses)[0]

if __name__=='__main__':

    __DEBGU__ = False
    if __DEBGU__:
        import pdb
        pdb.set_trace()

    train()



    

