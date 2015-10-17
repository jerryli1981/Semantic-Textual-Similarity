import optparse
import cPickle as pickle

import sgd as optimizer
from rnn import RNN

import dependency_tree as tr
import time
import numpy as np
import pdb
import collections
import random
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy
from multiprocessing import Pool

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


# roll all parameters into a single vector
def roll_params(params):
    nn.L, nn.WR, nn.WV, nn.b, nn.Ws, nn.bs = params
    return np.concatenate((nn.L.ravel(), nn.WR.ravel(), nn.WV.ravel(), nn.b.ravel(), nn.Ws.ravel(), nn.bs.ravel()))

def softmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)

    return x

def forwardProp(hparams, params, tree):
        #because many training iterations. 
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
                curr.hAct= np.tanh(WV.dot(curr.vec) + b)
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
                        rel_vec = WR[rel.index]
                        sum += rel_vec.dot(tree.nodes[i].hAct) 

                    curr.hAct = np.tanh(sum + WV.dot(curr.vec) + b)
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        # compute target distribution   
        if tree.score == 5.0:
            tree.score = 4.99999 
        
        target_distribution = np.zeros((1, outputDim))
        for i in xrange(outputDim):
            score_floor = np.floor(tree.score)
            if i == score_floor + 1:
                target_distribution[0,i] = tree.score - score_floor
            elif i == score_floor:
                target_distribution[0,i] = score_floor - tree.score + 1
            else:
                target_distribution[0,i] = 0  

        predicted_distribution= softmax((np.dot(Ws, tree.root.hAct) + bs).reshape(1, outputDim))
        tree.root.pd = predicted_distribution.reshape(outputDim)
        tree.root.td = target_distribution.reshape(outputDim)

        #cost = -np.dot(target_distribution, np.log(predicted_distribution).T)
        cost = entropy(target_distribution.reshape(outputDim,),predicted_distribution.reshape(outputDim,))

        #assert cost1[0,0] == cost2, "they should equal"
        
        #correctLabel = np.argmax(target_distribution)
        #guessLabel = np.argmax(predicted_distribution)
        #predictScore = predicted_distribution.reshape(outputDim,).dot(np.array([1,2,3,4,5]))
        #return cost, predictScore, tree.score
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

            dL[curr.index] += curr.deltas

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

def objAndGrad(par_data): 

    hparams, params = par_data[0]
    mbdata = par_data[1]

    relNum, wvecDim, outputDim, numWords = hparams

    cost = 0.0

    # Gradients
    dWR = np.empty((relNum, wvecDim, wvecDim))
    dWV = np.empty((wvecDim, wvecDim))
    db = np.empty((wvecDim))
    dWs = np.empty((outputDim,wvecDim))
    dbs = np.empty((outputDim))

    # Zero gradients
    dWR[:] = 0
    dWV[:] = 0
    db[:] = 0
    dWs[:] = 0
    dbs[:] = 0

    #create with default_factory : self.defaultVec = lambda : np.zeros((wvecDim,))
    defaultVec = lambda : np.zeros((wvecDim,))
    dL = collections.defaultdict(defaultVec)

    grads = (dL, dWR, dWV, db, dWs, dbs)

    params = unroll_params(params, hparams)

    for tree in mbdata: 
        c = forwardProp(hparams, params, tree)
        cost += c
        

    # Back prop each tree in minibatch
    for tree in mbdata:
        backProp(grads, params, tree)
        
    return cost

def par_objective(num_proc, data, hparams, params):

    pool = Pool(processes = num_proc)

    oparams = (hparams, params)

    n = len(data) / num_proc
    split_data = [data[i:i+n] for i in range(0, len(data), n)]
    to_map = []

    for item in split_data:
        to_map.append((oparams, item))

    result = pool.map(objAndGrad,to_map)
    pool.close()
    pool.join()



if __name__=='__main__':

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
    
    args = None

    (opts, args)=parser.parse_args()

    opts.numWords = len(tr.loadWordMap())

    opts.relNum = len(tr.loadRelMap())
    
    word2vecs = tr.loadWord2VecMap()

    num_proc = 4

    for c in xrange(opts.crossValidation):

        print "CV: %s"%c

        trainTrees = tr.loadTrees(c, "train")
        devTrees = tr.loadTrees(c, "dev")
        print "train size %s"%len(trainTrees)
        print "dev size %s"%len(devTrees)


        nn = RNN(opts.relNum, opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        
        nn.initParams(word2vecs)

        params = (nn.L, nn.WR, nn.WV, nn.b, nn.Ws, nn.bs)

        #grads = [nn.dWR, nn.dWV, nn.db, nn.dWs, nn.dbs]

        hparams = (nn.relNum, nn.wvecDim, nn.outputDim, nn.numWords)

        roll_params = roll_params(params)

        batch_size = 400
        
        for e in range(opts.epochs):
            print "Running epoch %d"%e

            random.shuffle(trainTrees)

            batches = [trainTrees[x : x + batch_size] for x in xrange(0, len(trainTrees), batch_size)]

            start = time.time()
            for batch in batches:
    
                par_objective(num_proc, batch, hparams,roll_params)
            
            end = time.time()
            print "Time per epoch : %f"%(end-start)


