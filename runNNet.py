import optparse
import cPickle as pickle

import sgd as optimizer

from rnn import RNN
import dependency_tree as tr
import time
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# This is the main training function of the codebase. You are intended to run this function via command line 
# or by ./run.sh

# You should update run.sh accordingly before you run it!


# TODO:
# Create your plots here

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
        
        nn.initParams()

        sgd = optimizer.SGD(nn,alpha=opts.step,minibatch=opts.minibatch,
            optimizer=opts.optimizer)
        
        for e in range(opts.epochs):
            start = time.time()
            print "Running epoch %d"%e
            sgd.run(trainTrees)
            end = time.time()
            #print "Time per epoch : %f"%(end-start)

            with open(opts.outFile,'w') as fid:
                pickle.dump(opts,fid)
                pickle.dump(sgd.costt,fid)
                nn.toFile(fid)
            if evaluate_accuracy_while_training:

                #print "testing on training set real quick"
                #train_accuracies.append(test(opts.outFile,"train",opts.model,trainTrees))
                #print "testing on dev set real quick"
                #dev_accuracies.append(test(opts.outFile,"dev",opts.model,devTrees))
                evl = test(opts.outFile,"dev",opts.model,devTrees)
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


def test(netFile,dataSet, model='RNN', trees=None):
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
        
        nn.initParams()
        nn.fromFile(fid)

    #print "Testing %s..."%model
    cost, grad, correct, guess= nn.costAndGrad(trees,test=True)
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


