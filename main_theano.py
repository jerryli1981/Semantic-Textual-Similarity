from Theano import *

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default=None)
    parser.add_argument("--epochs",dest="epochs",type=int,default=50)
    parser.add_argument("--step",dest="step",type=float,default=1e-2)
    parser.add_argument("--outputDim",dest="outputDim",type=int,default=5)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--wvecDim",dest="wvecDim",type=int,default=30)
    parser.add_argument("--outFile",dest="outFile",type=str, default="models/test.bin")
    parser.add_argument("--numProcess",dest="numProcess",type=int,default=None)
    parser.add_argument("--repModel",dest="repModel",type=str,default="lstm")
    parser.add_argument("--debug",dest="debug",type=str,default="False")
    parser.add_argument("--useLearnedModel",dest="useLearnedModel",type=str,default="False")
    args = parser.parse_args()

    if args.debug == "True":
        import pdb
        pdb.set_trace()

    
    import dependency_tree as tr     
    trainTrees = tr.loadTrees("train")
    devTrees = tr.loadTrees("dev")
    print "train number %d"%len(trainTrees)
    print "dev number %d"%len(devTrees)

    from optimization import Optimization


    rng = np.random.RandomState(1234)

    rep_model = initial_RepModel(rng, tr, args.repModel, args.wvecDim, args.outFile, args.useLearnedModel)

    rnn_optimizer = RNN_Optimization(rep_model, alpha=args.step, optimizer=args.optimizer)

    print "training model"
    train_with_theano_mlp(rng, rnn_optimizer, rep_model, trainTrees, devTrees, args.minibatch, args.epochs,
     args.hiddenDim, args.outputDim, args.step, args.optimizer, args.useLearnedModel, args.outFile, action="test")



    

    



