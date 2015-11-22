import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath('../Lasagne'))

from lasagne.layers import InputLayer, LSTMLayer, NonlinearityLayer, SliceLayer, FlattenLayer, EmbeddingLayer,\
    ElemwiseMergeLayer, AbsLayer,ReshapeLayer, get_output, get_all_params, get_output_shape, DropoutLayer,\
    DenseLayer,ElemwiseSumLayer,Conv2DLayer, CustomRecurrentLayer, \
    ConcatLayer, Pool1DLayer, FeaturePoolLayer,count_params

from lasagne.regularization import regularize_layer_params_weighted, l2, l1,regularize_layer_params,\
                                    regularize_network_params
from lasagne.nonlinearities import tanh, sigmoid, softmax, rectify
from lasagne.objectives import categorical_crossentropy, squared_error, categorical_accuracy
from lasagne.updates import sgd, adagrad, adadelta, nesterov_momentum, rmsprop, adam

from utils import read_sequence_dataset, iterate_minibatches,loadWord2VecMap


def build_network_ACL15(args, input1_var, input1_mask_var, 
        input2_var, intut2_mask_var, target_var, wordEmbeddings, maxlen=36):

    print("Building model ...")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    GRAD_CLIP = wordDim
    l1_in = InputLayer((None, maxlen),input_var=input1_var)
    batchsize, seqlen = l1_in.input_var.shape
    l1_mask_in = InputLayer((None, maxlen),input_var=input1_mask_var)
    l1_emb = EmbeddingLayer(l1_in, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    l1_emb.params[l1_emb.W].remove('trainable')

    l_forward_1_lstm = LSTMLayer(
        l1_emb, num_units=args.lstmDim, mask_input=l1_mask_in, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)


    """
    l_forward_1_b = LSTMLayer(
        l1_emb, num_units=args.lstmDim, mask_input=l1_mask_in, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh, backwards=True)
    """
    l_forward_1 = SliceLayer(l_forward_1_lstm, indices=-1, axis=1) # out_shape (None, args.lstmDim)
    """
    l_forward_1_b = SliceLayer(l_forward_1_b, indices=0, axis=1) # out_shape (None, args.lstmDim)

    l_forward_1 = ConcatLayer([l_forward_1, l_forward_1_b])
    """

    """
    l_forward_1 = SliceLayer(l_forward_1, indices=slice(-maxlen, None), axis=1)
    l_forward_1 = FeaturePoolLayer(l_forward_1, pool_size=maxlen, axis=1, pool_function=T.mean)
    l_forward_1 = ReshapeLayer(l_forward_1, ((batchsize, args.lstmDim)))
    """

    l2_in = InputLayer((None, maxlen),input_var=input2_var)
    l2_mask_in = InputLayer((None, maxlen),input_var=input2_mask_var)
    l2_emb = EmbeddingLayer(l2_in, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    l2_emb.params[l2_emb.W].remove('trainable')

    l_forward_2_lstm = LSTMLayer(
        l2_emb, num_units=args.lstmDim, mask_input=l2_mask_in, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)

    """"
    l_forward_2_b = LSTMLayer(
        l2_emb, num_units=args.lstmDim, mask_input=l2_mask_in, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh, backwards=True)
    """
    l_forward_2 = SliceLayer(l_forward_2_lstm, indices=-1, axis=1)

    """
    l_forward_2_b = SliceLayer(l_forward_2_b, indices=0, axis=1)
    l_forward_2 = ConcatLayer([l_forward_2, l_forward_2_b])

    """
    """
    l_forward_2 = SliceLayer(l_forward_2, indices=slice(-maxlen, None), axis=1)
    l_forward_2 = FeaturePoolLayer(l_forward_2, pool_size=maxlen, axis=1, pool_function=T.mean)
    l_forward_2 = ReshapeLayer(l_forward_2, ((batchsize, args.lstmDim)))
    """

    l12_mul = ElemwiseMergeLayer([l_forward_1, l_forward_2], merge_function=T.mul)
    l12_sub = AbsLayer(ElemwiseMergeLayer([l_forward_1, l_forward_2], merge_function=T.sub))
    l12_concat = ConcatLayer([l12_mul, l12_sub])

    #l12_concat = DropoutLayer(l12_concat, p=0.2)

    l_hid = DenseLayer(l12_concat, num_units=args.hiddenDim, nonlinearity=sigmoid)


    if args.task == "sts":
        network = DenseLayer(
                l_hid, num_units=5,nonlinearity=softmax)

    elif args.task == "ent":
        network = DenseLayer(
                l_hid, num_units=3,nonlinearity=softmax)

    prediction = get_output(network)
    loss = T.mean(target_var * ( T.log(target_var+ 1e-16) - T.log(prediction) ))
    #loss = T.mean(categorical_crossentropy(prediction, target_var))
    #loss += 0.0001 * sum (T.sum(layer_params ** 2) for layer_params in get_all_params(network) )
    #penalty = sum ( T.sum(lstm_param**2) for lstm_param in lstm_params )
    #penalty = regularize_layer_params(l_forward_1_lstm, l2)
    #penalty = T.sum(lstm_param**2 for lstm_param in lstm_params)
    #penalty = 0.0001 * sum (T.sum(layer_params ** 2) for layer_params in get_all_params(l_forward_1) )
    lambda_val = 0.5 * 1e-4

    layers = {l_forward_1_lstm:lambda_val, l_hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)
    loss = loss + penalty

    params = get_all_params(network, trainable=True)

    if args.optimizer == "sgd":
        updates = sgd(loss, params, learning_rate=args.step)
    elif args.optimizer == "adagrad":
        updates = adagrad(loss, params, learning_rate=args.step)
    elif args.optimizer == "adadelta":
        updates = adadelta(loss, params, learning_rate=args.step)
    elif args.optimizer == "nesterov":
        updates = nesterov_momentum(loss, params, learning_rate=args.step)
    elif args.optimizer == "rms":
        updates = rmsprop(loss, params, learning_rate=args.step)
    elif args.optimizer == "adam":
        updates = adam(loss, params, learning_rate=args.step)
    else:
        raise "Need set optimizer correctly"
 
    test_prediction = get_output(network, deterministic=True)
    #test_loss = T.mean(categorical_crossentropy(test_prediction, target_var))
    #test_loss = 0.2 * T.sum( target_var * ( T.log(target_var+ 1e-16) - T.log(test_prediction)))/ batchsize
    test_loss = T.mean(target_var * ( T.log(target_var+ 1e-16) - T.log(test_prediction) ))

    train_fn = theano.function([input1_var, input1_mask_var, input2_var, intut2_mask_var, target_var], 
        loss, updates=updates, allow_input_downcast=True)

    if args.task == "sts":
        val_fn = theano.function([input1_var, input1_mask_var, input2_var, intut2_mask_var, target_var], 
            [test_loss, test_prediction], allow_input_downcast=True)

    elif args.task == "ent":
        #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
        test_acc = T.mean(categorical_accuracy(test_prediction, target_var))

        val_fn = theano.function([input1_var, input1_mask_var, input2_var, intut2_mask_var, target_var], 
            [test_loss, test_acc], allow_input_downcast=True)

    return train_fn, val_fn


def build_network(args, input1_var=None, input2_var=None, target_var=None, maxlen=36):

    print("Building model ...")

    l1_in = InputLayer((None, maxlen, args.wvecDim),input_var=input1_var)

    l2_in = InputLayer((None, maxlen, args.wvecDim),input_var=input2_var)

    batchsize, seqlen, _ = l1_in.input_var.shape

    GRAD_CLIP = args.wvecDim

    #l1_in = ReshapeLayer(l1_in,(-1, maxlen, args.wvecDim))
    l_forward_1 = LSTMLayer(
        l1_in, num_units=args.wvecDim, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_1 = lasagne.layers.SliceLayer(l_forward_1, indices=slice(-3, None), axis=1)

    l_forward_1 = ReshapeLayer(l_forward_1,(batchsize, 1, 3, args.wvecDim))

    #l2_in = ReshapeLayer(l2_in,(-1, maxlen, args.wvecDim))
    l_forward_2 = LSTMLayer(
        l2_in, args.wvecDim, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.SliceLayer(l_forward_2, indices=slice(-3, None), axis=1)

    l_forward_2 = ReshapeLayer(l_forward_2,(batchsize, 1, 3, args.wvecDim))


    l_forward_1 = Conv2DLayer(
            l_forward_1, num_filters=32, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
 
    # Max-pooling layer of factor 2 in both dimensions:
    l_forward_1 = lasagne.layers.MaxPool2DLayer(l_forward_1, pool_size=(2, 2))

    """"
    #another con2d
    l_forward_1 = lasagne.layers.Conv2DLayer(
            l_forward_1, num_filters=32, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify)

    l_forward_1 = lasagne.layers.MaxPool2DLayer(l_forward_1, pool_size=(2, 2))
    """

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    l_forward_1 = DenseLayer(
            lasagne.layers.dropout(l_forward_1, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_forward_2 = Conv2DLayer(
            l_forward_2, num_filters=32, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
 
    # Max-pooling layer of factor 2 in both dimensions:
    l_forward_2 = lasagne.layers.MaxPool2DLayer(l_forward_2, pool_size=(2, 2))

    """"
    #another con2d
    l_forward_2 = lasagne.layers.Conv2DLayer(
            l_forward_2, num_filters=32, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify)
    l_forward_2 = lasagne.layers.MaxPool2DLayer(l_forward_2, pool_size=(2, 2))
    """

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    l_forward_2 = DenseLayer(
            lasagne.layers.dropout(l_forward_2, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)


    #l_forward_1 = FlattenLayer(l_forward_1)

    #l_forward_2 = FlattenLayer(l_forward_2)

    #l_forward_1 = lasagne.layers.SliceLayer(l_forward_1, indices=-1, axis=1)

    #l_forward_2 = lasagne.layers.SliceLayer(l_forward_2, indices=-1, axis=1)

    # elementwisemerge need fix the sequence length
    l12_mul = ElemwiseMergeLayer([l_forward_1, l_forward_2], merge_function=T.mul)
    l12_sub = ElemwiseMergeLayer([l_forward_1, l_forward_2], merge_function=T.sub)
    l12_sub = AbsLayer(l12_sub)

    #l12_mul  = ReshapeLayer(l12_mul,(-1, args.wvecDim))
    #l12_sub = ReshapeLayer(l12_sub,(-1, args.wvecDim))

    l12_mul_Dense = DenseLayer(l12_mul, num_units=args.hiddenDim, nonlinearity=None, b=None)
    l12_sub_Dense = DenseLayer(l12_sub, num_units=args.hiddenDim, nonlinearity=None, b=None)


    #l12_mul_Dense_r = ReshapeLayer(l12_mul_Dense, (batchsize, seqlen, args.hiddenDim))
    #l12_sub_Dense_r  = ReshapeLayer(l12_sub_Dense, (batchsize, seqlen, args.hiddenDim))
    
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

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=20)
    parser.add_argument("--step",dest="step",type=float,default=0.01)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--lstmDim",dest="lstmDim",type=int,default=30)
    parser.add_argument("--task",dest="task",type=str,default=None)
    args = parser.parse_args()


    # Load the dataset
    print("Loading data...")
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    X1_train, X1_mask_train, X2_train, X2_mask_train, Y_labels_train, Y_scores_train, Y_scores_pred_train = \
        read_sequence_dataset(sick_dir, "train")
    X1_dev, X1_mask_dev, X2_dev, X2_mask_dev, Y_labels_dev, Y_scores_dev, Y_scores_pred_dev = \
        read_sequence_dataset(sick_dir, "dev")
    X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, Y_scores_test, Y_scores_pred_test = \
        read_sequence_dataset(sick_dir, "test")

    input1_var = T.imatrix('inputs_1')
    input2_var = T.imatrix('inputs_2')
    input1_mask_var = T.matrix('inputs_mask_1')
    input2_mask_var = T.matrix('inputs_mask_2')
    target_var = T.fmatrix('targets')

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings.astype(np.float32)

    train_fn, val_fn = build_network_ACL15(args, input1_var, input1_mask_var, input2_var, input2_mask_var,
        target_var, wordEmbeddings)

    print("Starting training...")
    best_val_acc = 0
    best_val_pearson = 0
    for epoch in range(args.epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X1_train, X1_mask_train, X2_train, X2_mask_train, Y_labels_train,
            Y_scores_train, Y_scores_pred_train, args.minibatch, shuffle=True):

            inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

            if args.task == "sts":
                train_err += train_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, scores_pred)
            elif args.task == "ent":
                train_err += train_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, labels)
            else:
                raise "task need to be set"

            train_batches += 1
 
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_pearson = 0

        for batch in iterate_minibatches(X1_dev, X1_mask_dev, X2_dev, X2_mask_dev, Y_labels_dev, Y_scores_dev, 
            Y_scores_pred_dev, len(X1_dev), shuffle=False):

            inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

            if args.task == "sts":

                err, preds = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, scores_pred)
                predictScores = preds.dot(np.array([1,2,3,4,5]))
                guesses = predictScores.tolist()
                scores = scores.tolist()
                pearson_score = pearsonr(scores,guesses)[0]
                val_pearson += pearson_score 

            elif args.task == "ent":
                err, acc = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, labels)
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
    for batch in iterate_minibatches(X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, 
        Y_scores_test, Y_scores_pred_test, len(X1_test), shuffle=False):

        inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

        if args.task == "sts":

            err, preds = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, scores_pred)
            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            test_pearson += pearson_score 

        elif args.task == "ent":
            err, acc = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, labels)
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