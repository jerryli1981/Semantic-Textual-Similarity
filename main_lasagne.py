import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath('../Lasagne'))
#sys.path.append('../Lasagne')

import lasagne

def load_data_matrix(data, args, seq_len=36, n_children=6, unfinished_flag=-2):

    Y = np.zeros((len(data), args.outputDim+1), dtype=np.float32)
    scores = np.zeros((len(data)), dtype=np.float32)

    # to store hidden representation
    #(rootFlag, finishedFlag, globalgovIdx, n_children* (locaDepIdx, globalDepIdx, relIdx) , hiddenRep)
    storage_dim = 1 + 1 + 1 + 3*n_children + args.wvecDim

    X1 = np.zeros((len(data), seq_len, storage_dim), dtype=np.float32)
    X1.fill(-1.0)
    X2 = np.zeros((len(data), seq_len, storage_dim), dtype=np.float32)
    X2.fill(-1.0)
    
    for i, (score, item) in enumerate(data):
        first_t, second_t= item

        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y[i, floor] = 1
        else:
            Y[i, floor] = ceil-sim
            Y[i, ceil] = sim-floor

        f_idxSet = set()
        for govIdx, depIdx in first_t.dependencies:
            f_idxSet.add(govIdx)
            f_idxSet.add(depIdx)

        for j, Node in enumerate(first_t.nodes):

            if j not in f_idxSet:
                continue

            node_vec = np.zeros((storage_dim,), dtype=np.float32)
            node_vec.fill(-1.0)
            if j == first_t.rootIdx:
                node_vec[0] = 1

            node_vec[1] = unfinished_flag
            node_vec[2] = Node.index

            if len(Node.kids) != 0:

                r = range(0, 3*n_children, 3)
                r = r[:len(Node.kids)]
                for d, c in enumerate(r):
                    localDepIdx, rel = Node.kids[d]
                    node_vec[3+c] = localDepIdx
                    node_vec[4+c] = first_t.nodes[localDepIdx].index
                    node_vec[5+c] = rel.index


            X1[i, j] = node_vec


        s_idxSet = set()
        for govIdx, depIdx in second_t.dependencies:
            s_idxSet.add(govIdx)
            s_idxSet.add(depIdx)

        for j, Node in enumerate(second_t.nodes):

            if j not in s_idxSet:
                continue

            node_vec = np.zeros((storage_dim,), dtype=np.float32)
            node_vec.fill(-1.0)
            if j == second_t.rootIdx:
                node_vec[0] = 1

            node_vec[1] = unfinished_flag
            node_vec[2] = Node.index

            if len(Node.kids) != 0:

                r = range(0, 3*n_children, 3)
                r = r[:len(Node.kids)]
                for d, c in enumerate(r):
                    localDepIdx, rel = Node.kids[d]
                    node_vec[3+c] = localDepIdx
                    node_vec[4+c] = second_t.nodes[localDepIdx].index
                    node_vec[5+c] = rel.index

            X2[i, j] = node_vec
   
        scores[i] = score

    Y = Y[:, 1:]

    input_shape = (len(data), seq_len, storage_dim)
      
    return X1, X2, Y, scores, input_shape

def load_data(data, wordEmbeddings, args, maxlen=36):

    Y_scores_pred = np.zeros((len(data), args.rangeScores+1), dtype=np.float32)

    #maxlen = 0
    for i, (label, score, l_t, r_t) in enumerate(data):
        """
        max_ = max(len(l_t.nodes), len(r_t.nodes))
        if maxlen < max_:
            maxlen = max_
        """
        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y_scores_pred[i, floor] = 1
        else:
            Y_scores_pred[i, floor] = ceil-sim 
            Y_scores_pred[i, ceil] = sim-floor

    Y_scores_pred = Y_scores_pred[:, 1:]

    X1 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)
    X2 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)

    Y_labels = np.zeros((len(data)), dtype=np.int32)
    Y_scores = np.zeros((len(data)), dtype=np.float32)

    #np.random.uniform(-0.25,0.25,k)

    for i, (label, score, l_tree, r_tree) in enumerate(data):

        for j, Node in enumerate(l_tree.nodes):
            X1[i, j] =  wordEmbeddings[:, Node.index]
            if j >= len(l_tree.nodes):
                X1[i,j] =  np.random.uniform(-0.25,0.25, args.wvecDim)

        for k, Node in enumerate(r_tree.nodes):
            X2[i, k] =  wordEmbeddings[:, Node.index]
            if j >= len(r_tree.nodes):
                X2[i, j] =  np.random.uniform(-0.25,0.25, args.wvecDim)

        Y_labels[i] = label
        Y_scores[i] = score
        
    return X1, X2, Y_labels, Y_scores, Y_scores_pred


def build_network(args, input1_var=None, input2_var=None, target_var=None, maxlen=36):

    print("Building model 0 and compiling functions...")

    from lasagne.layers import InputLayer, LSTMLayer, NonlinearityLayer, SliceLayer, FlattenLayer,\
    ElemwiseMergeLayer, AbsLayer,ReshapeLayer,DenseLayer,ElemwiseSumLayer,Conv2DLayer, CustomRecurrentLayer
    from lasagne.regularization import regularize_layer_params_weighted, l2, l1,regularize_layer_params

    l1_in = InputLayer((None, maxlen, args.wvecDim),input_var=input1_var)

    l2_in = InputLayer((None, maxlen, args.wvecDim),input_var=input2_var)

    batchsize, seqlen, _ = l1_in.input_var.shape

    GRAD_CLIP = args.wvecDim

    #l1_in = ReshapeLayer(l1_in,(-1, maxlen, args.wvecDim))
    l_forward_1 = LSTMLayer(
        l1_in, num_units=args.wvecDim, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_1 = ReshapeLayer(l_forward_1,(batchsize, 1, maxlen, args.wvecDim))

    #l2_in = ReshapeLayer(l2_in,(-1, maxlen, args.wvecDim))
    l_forward_2 = LSTMLayer(
        l2_in, args.wvecDim, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    l_forward_2 = ReshapeLayer(l_forward_2,(batchsize, 1, maxlen, args.wvecDim))


    l_forward_1 = Conv2DLayer(
            l_forward_1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
 
    # Max-pooling layer of factor 2 in both dimensions:
    l_forward_1 = lasagne.layers.MaxPool2DLayer(l_forward_1, pool_size=(2, 2))

    #another con2d
    l_forward_1 = lasagne.layers.Conv2DLayer(
            l_forward_1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    l_forward_1 = lasagne.layers.MaxPool2DLayer(l_forward_1, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    l_forward_1 = DenseLayer(
            lasagne.layers.dropout(l_forward_1, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_forward_2 = Conv2DLayer(
            l_forward_2, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
 
    # Max-pooling layer of factor 2 in both dimensions:
    l_forward_2 = lasagne.layers.MaxPool2DLayer(l_forward_2, pool_size=(2, 2))

    #another con2d
    l_forward_2 = lasagne.layers.Conv2DLayer(
            l_forward_2, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    l_forward_2 = lasagne.layers.MaxPool2DLayer(l_forward_2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    l_forward_2 = DenseLayer(
            lasagne.layers.dropout(l_forward_2, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    # elementwisemerge need fix the sequence length
    l12_mul = ElemwiseMergeLayer([l_forward_1, l_forward_2], merge_function=T.mul)
    l12_sub = ElemwiseMergeLayer([l_forward_1, l_forward_2], merge_function=T.sub)
    l12_sub = AbsLayer(l12_sub)


    l12_mul_Dense = DenseLayer(l12_mul, num_units=args.hiddenDim, nonlinearity=None, b=None)
    l12_sub_Dense = DenseLayer(l12_sub, num_units=args.hiddenDim, nonlinearity=None, b=None)
    
    joined = ElemwiseSumLayer([l12_mul_Dense, l12_sub_Dense])

    l_hid = NonlinearityLayer(joined, nonlinearity=lasagne.nonlinearities.sigmoid)

    if args.task == "sts":
        network = lasagne.layers.DenseLayer(
                l_hid, num_units=args.rangeScores,
                nonlinearity=lasagne.nonlinearities.softmax)

    elif args.task == "ent":
        network = DenseLayer(
                l_hid, num_units=args.numLabels,
                nonlinearity=lasagne.nonlinearities.softmax)

    prediction = lasagne.layers.get_output(network)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(prediction, target_var))

    #layers = {l12_mul_Dense:0.1, l12_sub_Dense:0.1, l_out_Dense:0.5}

    #l2_penalty = regularize_layer_params_weighted(layers, l2)
    #l1_penalty = regularize_layer_params(l_out_Dense, l1) * 1e-4
    #loss = loss + l2_penalty + l1_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)

    if args.optimizer == "sgd":
        updates = lasagne.updates.sgd(loss, params, learning_rate=args.step)
    elif args.optimizer == "adagrad":
        updates = lasagne.updates.adagrad(loss, params, learning_rate=args.step)
    elif args.optimizer == "adadelta":
        updates = lasagne.updates.adadelta(loss, params, learning_rate=args.step)
    elif args.optimizer == "nesterov":
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=args.step)
    elif args.optimizer == "rms":
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=args.step)
    else:
        raise "Need set optimizer correctly"
 

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([input1_var, input2_var, target_var], loss, 
        updates=updates, allow_input_downcast=True)

    if args.task == "sts":
        val_fn = theano.function([input1_var, input2_var, target_var], 
            [test_loss, test_prediction], allow_input_downcast=True)

    elif args.task == "ent":
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        val_fn = theano.function([input1_var, input2_var, target_var], 
            [test_loss, test_acc], allow_input_downcast=True)

    return train_fn, val_fn

def iterate_minibatches(inputs1, inputs2, targets, scores, scores_pred, batchsize, shuffle=False):
    assert len(inputs1) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs1[excerpt], inputs2[excerpt], targets[excerpt], scores[excerpt], scores_pred[excerpt]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=20)
    parser.add_argument("--step",dest="step",type=float,default=0.01)
    parser.add_argument("--rangeScores",dest="rangeScores",type=int,default=5)
    parser.add_argument("--numLabels",dest="numLabels",type=int,default=3)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--wvecDim",dest="wvecDim",type=int,default=30)
    parser.add_argument("--mlpActivation",dest="mlpActivation",type=str,default="sigmoid")
    parser.add_argument("--task",dest="task",type=str,default=None)
    args = parser.parse_args()


    # Load the dataset
    print("Loading data...")
    import dependency_tree as tr     
    trainTrees = tr.loadTrees("train")
    devTrees = tr.loadTrees("dev")
    testTrees = tr.loadTrees("test")
    
    print "train number %d"%len(trainTrees)
    print "dev number %d"%len(devTrees)
    print "test number %d"%len(testTrees)

    wordEmbeddings = tr.loadWord2VecMap()[:args.wvecDim, :]
    
    X1_train, X2_train, Y_labels_train, Y_scores_train, Y_scores_pred_train = load_data(trainTrees, wordEmbeddings, args)
    X1_dev, X2_dev, Y_labels_dev, Y_scores_dev, Y_scores_pred_dev = load_data(devTrees, wordEmbeddings, args)
    X1_test, X2_test, Y_labels_test, Y_scores_test, Y_scores_pred_test = load_data(testTrees, wordEmbeddings, args)

    input1_var = T.tensor3('inputs_1')
    input2_var = T.tensor3('inputs_2')

    if args.task=="sts":
        target_var = T.fmatrix('targets')
    elif args.task=="ent":
        target_var = T.ivector('targets')

    train_fn, val_fn = build_network(args, input1_var, input2_var, target_var)

    print("Starting training...")
    best_val_acc = 0
    best_val_pearson = 0
    for epoch in range(args.epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X1_train, X2_train, Y_labels_train,
            Y_scores_train, Y_scores_pred_train, args.minibatch, shuffle=True):

            inputs1, inputs2, labels, scores, scores_pred = batch

            if args.task == "sts":
                train_err += train_fn(inputs1, inputs2, scores_pred)
            elif args.task == "ent":
                train_err += train_fn(inputs1, inputs2, labels)
            else:
                raise "task need to be set"

            train_batches += 1
 
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_pearson = 0

        for batch in iterate_minibatches(X1_dev, X2_dev, Y_labels_dev, Y_scores_dev, 
            Y_scores_pred_dev, len(X1_dev), shuffle=False):

            inputs1, inputs2, labels, scores, scores_pred = batch

            if args.task == "sts":

                err, preds = val_fn(inputs1, inputs2, scores_pred)
                predictScores = preds.dot(np.array([1,2,3,4,5]))
                guesses = predictScores.tolist()
                scores = scores.tolist()
                pearson_score = pearsonr(scores,guesses)[0]
                val_pearson += pearson_score 

            elif args.task == "ent":
                err, acc = val_fn(inputs1, inputs2, labels)
                val_acc += acc

            val_err += err
            
            val_batches += 1

            
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


        if args.task == "sts":
            val_score = val_pearson / val_batches * 100
            print("  validation pearson:\t\t{:.2f} %".format(
                val_pearson / val_batches * 100))
            if best_val_pearson < val_score:
                best_val_pearson = val_score

        elif args.task == "ent":
            val_score = val_acc / val_batches * 100
            print("  validation accuracy:\t\t{:.2f} %".format(val_score))
            if best_val_acc < val_score:
                best_val_acc = val_score

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_pearson = 0
    test_batches = 0
    for batch in iterate_minibatches(X1_test, X2_test, Y_labels_test, 
        Y_scores_test, Y_scores_pred_test, len(X1_test), shuffle=False):

        inputs1, inputs2, labels, scores, scores_pred = batch

        if args.task == "sts":

            err, preds = val_fn(inputs1, inputs2, scores_pred)
            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            test_pearson += pearson_score 

        elif args.task == "ent":
            err, acc = val_fn(inputs1, inputs2, labels)
            test_acc += acc


        test_err += err
        
        test_batches += 1


    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    if args.task == "sts":
        print("  Best validate perason:\t\t{:.2f} %".format(best_val_pearson))
        print("  test pearson:\t\t{:.2f} %".format(
            test_pearson / test_batches * 100))
    elif args.task == "ent":
        print("  Best validate accuracy:\t\t{:.2f} %".format(best_val_acc))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))



    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)