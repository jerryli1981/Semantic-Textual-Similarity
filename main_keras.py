import sys
import os
import time

import numpy as np
import theano

from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath('../keras'))

from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2,activity_l2


def load_data(data, args):

    Y_scores_pred = np.zeros((len(data), args.rangeScores+1), dtype=np.float32)

    maxlen = 0
    for i, (label, score, l_t, r_t) in enumerate(data):

        max_ = max(len(l_t.nodes), len(r_t.nodes))
        if maxlen < max_:
            maxlen = max_

        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y_scores_pred[i, floor] = 1
        else:
            Y_scores_pred[i, floor] = ceil-sim 
            Y_scores_pred[i, ceil] = sim-floor

    Y_scores_pred = Y_scores_pred[:, 1:]

    Y_scores = np.zeros((len(data)), dtype=np.float32)
    labels = []
    X1 = []
    X2 =[]
    for label, score, l_tree, r_tree in data:

        l_n_idx = []
        for Node in l_tree.nodes:
            l_n_idx.append(Node.index)

        X1.append(l_n_idx)

        r_n_idx = []
        for Node in r_tree.nodes:
            r_n_idx.append(Node.index)

        X2.append(r_n_idx)

        labels.append(label)
        Y_scores[i] = score

    Y_labels = np.zeros((len(labels), args.numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    X1 = sequence.pad_sequences(X1, maxlen=maxlen)
    X2 = sequence.pad_sequences(X2, maxlen=maxlen)   
        
    return X1, X2, Y_labels, Y_scores, Y_scores_pred

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

def build_network(args, wordEmbeddings, maxlen=36, reg=1e-4):
 
    print("Building model and compiling functions...")
    n_symbols = wordEmbeddings.shape[1]
    wordEmbeddings = wordEmbeddings[:args.wvecDim, :]

    l_lstm_1 = Sequential()
    l_lstm_1.add(Embedding(input_dim=n_symbols, output_dim=args.wvecDim, 
        mask_zero=True, weights=[wordEmbeddings.T],input_length=maxlen))
    l_lstm_1.add(LSTM(output_dim=args.wvecDim, return_sequences=False, 
        input_shape=(maxlen, args.wvecDim)))
    l_lstm_1.add(Dropout(0.1))
    l_lstm_1.layers[1].regularizers = [l2(reg)] * 12
    for i in range(12):    
        l_lstm_1.layers[1].regularizers[i].set_param(l_lstm_1.layers[1].get_params()[0][i])


    l_lstm_2 = Sequential()
    l_lstm_2.add(Embedding(input_dim=n_symbols, output_dim=args.wvecDim, 
        mask_zero=True, weights=[wordEmbeddings.T],input_length=maxlen))
    l_lstm_2.add(LSTM(output_dim=args.wvecDim, return_sequences=False, 
        input_shape=(maxlen, args.wvecDim)))
    l_lstm_2.add(Dropout(0.1))
    l_lstm_2.layers[1].regularizers = [l2(reg)] * 12
    for i in range(12):    
        l_lstm_2.layers[1].regularizers[i].set_param(l_lstm_2.layers[1].get_params()[0][i])


    l_mul = Sequential()
    l_mul.add(Merge([l_lstm_1, l_lstm_2], mode='mul'))

    l_sub = Sequential()
    l_sub.add(Merge([l_lstm_1, l_lstm_2], mode='abs_sub'))

    model = Sequential()
    model.add(Merge([l_mul, l_sub], mode='concat', concat_axis=-1))
    model.add(Dense(output_dim=2*args.wvecDim,W_regularizer=l2(reg),b_regularizer=l2(reg)))

    if args.mlpActivation == "sigmoid":
        model.add(Activation('sigmoid'))
    elif args.mlpActivation == "tanh":
        model.add(Activation('tanh'))
    elif args.mlpActivation == "relu":
        model.add(Activation('relu'))

    if args.task=="sts":
        model.add(Dense(args.rangeScores,W_regularizer=l2(reg), b_regularizer=l2(reg)))
    elif args.task == "ent":
        model.add(Dense(args.numLabels,W_regularizer=l2(reg), b_regularizer=l2(reg)))

    model.add(Activation('softmax'))


    if args.optimizer == "sgd":
        optimizer = SGD(lr=args.step)
    elif args.optimizer == "adagrad":
        optimizer = Adagrad(lr=args.step)
    elif args.optimizer == "adadelta":
        optimizer = Adadelta(lr=args.step)
    elif args.optimizer == "rms":
        optimizer = RMSprop(lr=args.step)
    elif args.optimizer == "adam":
        optimizer = Adam(lr=args.step)
    else:
        raise "Need set optimizer correctly"

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    train_fn = model.train_on_batch
    val_fn = model.test_on_batch 
    predict_proba = model.predict_proba

    return train_fn, val_fn, predict_proba


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="sgd")
    parser.add_argument("--epochs",dest="epochs",type=int,default=5)
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

    X1_train, X2_train, Y_labels_train, Y_scores_train, Y_scores_pred_train = load_data(trainTrees, args)
    X1_dev, X2_dev, Y_labels_dev, Y_scores_dev, Y_scores_pred_dev = load_data(devTrees, args)
    X1_test, X2_test, Y_labels_test, Y_scores_test, Y_scores_pred_test = load_data(testTrees, args)


    wordEmbeddings = tr.loadWord2VecMap()
    train_fn, val_fn, predict_proba= build_network(args, wordEmbeddings)

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
                train_err += train_fn([inputs1, inputs2], scores_pred)
            elif args.task == "ent":
                train_err += train_fn([inputs1, inputs2], labels)
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

                err = val_fn([inputs1, inputs2], scores_pred)
                preds = predict_proba([inputs1, inputs2])
                predictScores = preds.dot(np.array([1,2,3,4,5]))
                guesses = predictScores.tolist()
                scores = scores.tolist()
                pearson_score = pearsonr(scores,guesses)[0]
                val_pearson += pearson_score 

            elif args.task == "ent":
                err, acc = val_fn([inputs1, inputs2], labels, accuracy=True)
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
            err = val_fn([inputs1, inputs2], scores_pred)
            preds = predict_proba([inputs1, inputs2])
            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            test_pearson += pearson_score 

        elif args.task == "ent":
            err, acc = val_fn([inputs1, inputs2], labels, accuracy=True)
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