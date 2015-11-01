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
    parser.add_argument("--activation",dest="activation",type=str,default=None)
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
     
    optimizer = Optimization(alpha=args.step, optimizer=args.optimizer)

    optimizer.initial_RepModel(tr, args.repModel, args.wvecDim, args.activation)

    optimizer.initial_theano_mlp(args.hiddenDim, args.outputDim, args.activation)


    best_dev_score  = 0.

    print "training model"
    for e in range(args.epochs):
        
        #print "Running epoch %d"%e
        optimizer.train_with_theano_mlp(trainTrees, args.minibatch)
        #print "Time per epoch : %f"%(end-start)
        cost, dev_score = optimizer.predict(devTrees)
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print "iter:%d cost: %f dev_score: %f best_dev_score %f"%(e, cost, dev_score, best_dev_score)
        else:
            print "iter:%d cost: %f dev_score: %f"%(e, cost, dev_score)



    



