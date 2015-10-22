import dependency_tree as tr
import time

from depTreeRNNModel import depTreeRnnModel as rnn

if __name__=='__main__':

    __DEBGU__ = False
    if __DEBGU__:
        import pdb
        pdb.set_trace()

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

    relNum = len(tr.loadRelMap())

    model = rnn(relNum, args.wvecDim, args.outputDim)
    
    word2vecs = tr.loadWord2VecMap()

    model.initialParams(word2vecs)

    trainTrees = tr.loadTrees("train")
    devTrees = tr.loadTrees("dev")

    best_dev_score  = 0.

    print "training model"
    for e in range(args.epochs):
        print "Running epoch %d"%e
        start = time.time()
        model.train(trainTrees, args.step, args.minibatch, numProcess=args.numProcess)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        dev_score = model.predict(devTrees)
        print "dev score %f", dev_score

        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print "best dev score %f", best_dev_score



    

