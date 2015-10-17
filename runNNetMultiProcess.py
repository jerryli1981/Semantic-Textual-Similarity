import optparse
import cPickle as pickle

import sgd as optimizer


from rnn import RNN

import dependency_tree as tr
import time
import matplotlib.pyplot as plt
import numpy as np
import pdb
import collections
import random
import copy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy
from multiprocessing import Pool

def softmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)

    return x

def forwardProp(hparams, params, tree):
        #because many training iterations. 
        tree.resetFinished()

        L, WR, WV, b, Ws, bs = params

        WV_shape, WR_shape, wvecDim, Ws_shape, outputDim = hparams

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
        
        correctLabel = np.argmax(target_distribution)
        guessLabel = np.argmax(predicted_distribution)
        predictScore = predicted_distribution.reshape(outputDim,).dot(np.array([1,2,3,4,5]))
        return cost, predictScore, tree.score

def backProp(grads, params, tree):

    dWR, dWV, db, dWs, dbs, dL = grads
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

def objAndGrad(data): 

        hparams = data[0]
        params = data[1]
        grads = data[2]
        mbdata = data[3]

        WV_shape, WR_shape, wvecDim, Ws_shape, outputDim = hparams

        cost = 0.0

    
        dWR, dWV, db, dWs, dbs = grads

        # Zero gradients
        dWV[:] = 0
        dWR[:] = 0
        db[:] = 0
        dWs[:] = 0
        dbs[:] = 0

        #create with default_factory : self.defaultVec = lambda : np.zeros((wvecDim,))
        defaultVec = lambda : np.zeros((wvecDim,))
        dL = collections.defaultdict(defaultVec)

        grads.append(dL)
        # Forward prop each tree in minibatch
        corrects = []
        guesses = []
        for tree in mbdata: 
            c, correct, guess = forwardProp(hparams, params, tree)
            corrects.append(correct)
            guesses.append(guess)
            cost += c

        # Back prop each tree in minibatch
        for tree in mbdata:
            backProp(grads, params, tree)

        # scale cost and grad by mb size
        scale = (1./len(mbdata))
        for v in dL.itervalues():
            v *=scale
        
        """
        # Add L2 Regularization 
        cost += (model.rho/2)*np.sum(model.WV**2)
        cost += (model.rho/2)*np.sum(model.WR**2)
        cost += (model.rho/2)*np.sum(model.Ws**2)

        return scale*cost,[dL,scale*(dWR + rho*WR),
                                   scale*(model.dWV + model.rho*model.WV),
                                   scale*model.db,
                                  scale*(model.dWs+model.rho*model.Ws),scale*model.dbs]
        """
        return 0,0

def par_objective(trainTrees,i,n,batch_size,hparams,params,grads):
    pool = Pool(processes = 4)
    trees = trainTrees[i:i+batch_size]
    split_data = [trees[j:j+n] for j in range(0,batch_size,n)]
    to_map = []
    for item in split_data:
        to_map.append((hparams, params, grads, item))

    result = pool.map(objAndGrad,to_map)
    pool.close()
    pool.join()

def run(args=None):
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

    (opts,args)=parser.parse_args(args)


    # make this false if you dont care about your accuracies per epoch, makes things faster!
    evaluate_accuracy_while_training = True

    # Testing
    if opts.test:
        test(opts.inFile,opts.data,opts.model)
        return

    opts.numWords = len(tr.loadWordMap())

    opts.relNum = len(tr.loadRelMap())
    
    word2vecs = tr.loadWord2VecMap()

    num_proc = 4

    
    
    for c in xrange(opts.crossValidation):

        train_accuracies = []
        dev_accuracies = []
        dev_pearsons = []
        dev_spearmans = []

        print "CV: %s"%c

        trainTrees = tr.loadTrees(c, "train")
        devTrees = tr.loadTrees(c, "dev")
        print "train size %s"%len(trainTrees)
        print "dev size %s"%len(devTrees)

        if(opts.model=='RNN'):
            nn = RNN(opts.relNum, opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        else:
            raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN'%opts.model
        
        nn.initParams(word2vecs)

        params = nn.stack
        grads = [nn.dWR, nn.dWV, nn.db, nn.dWs, nn.dbs]
        hparams = [nn.WV.shape, nn.WR.shape, nn.wvecDim, nn.Ws.shape, nn.outputDim]

        random.shuffle(trainTrees)

        #chunk size
        m = len(trainTrees)

        batch_size = 400

        n = batch_size/ num_proc
        
        for e in range(opts.epochs):
            start = time.time()
            print "Running epoch %d"%e
            
            for i in xrange(0,m-batch_size+1,batch_size):

                par_objective(trainTrees,i,n,batch_size,hparams,params,grads)

            end = time.time()
            print "Time per epoch : %f"%(end-start)
            
            """
            with open(opts.outFile,'w') as fid:
                pickle.dump(opts,fid)
                pickle.dump(sgd.costt,fid)
                nn.toFile(fid)
            if evaluate_accuracy_while_training:

                #print "testing on training set real quick"
                #train_accuracies.append(test(opts.outFile,"train",word2vecs,opts.model,trainTrees))
                #print "testing on dev set real quick"
                #dev_accuracies.append(test(opts.outFile,"dev",opts.model,devTrees))
                evl = test(opts.outFile,"dev",word2vecs,opts.model,devTrees)
                dev_pearsons.append(evl[0])
                dev_spearmans.append(evl[1])

                # because tesing need to forward propogation, so clear the fprop flags in trees and dev_trees
                for tree in trainTrees:
                    tree.resetFinished()
                for tree in devTrees:
                    tree.resetFinished()
                #print "fprop in trees cleared"

        if evaluate_accuracy_while_training:
            #train_errors = [ 1-acc for acc in train_accuracies]
            #dev_errors = [1-acc for acc in dev_accuracies]
            #print "train accuracies", train_accuracies
            #print "dev accuracies",dev_accuracies
            print "dev pearsons", dev_pearsons
            print "dev spearmanr", dev_spearmans
        """

def test(netFile,dataSet, word2vecs,model='RNN', trees=None):
    if trees==None:
        trees = tr.loadTrees(dataSet)

    assert netFile is not None, "Must give model to test"

    #print "Testing netFile %s"%netFile

    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        
        if(model=='RNN'):
            nn = RNN(opts.relNum, opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        else:
            raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN'%opts.model
        
        nn.initParams(word2vecs)
        nn.fromFile(fid)

    #print "Testing %s..."%model
    cost, grad, correct, guess= nn.costAndGrad(trees,test=True)
    print "Cost %f"%cost
    """
    correct_sum = 0
    total = len(trees)
    for i in xrange(0,len(correct)):        
        correct_sum+=(guess[i]==correct[i])
        
    print "Cost %f, Acc %f"%(cost,correct_sum/float(total))
    return correct_sum/float(total)
    """
    print "Pearson correlation %f"%(pearsonr(correct,guess)[0])
    print "Spearman correlation %f"%(spearmanr(correct,guess)[0])
    return pearsonr(correct,guess)[0],spearmanr(correct,guess)[0]


if __name__=='__main__':
    run()


