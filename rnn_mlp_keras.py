import sys
import os
import time

import numpy as np
import theano

from scipy.stats import pearsonr

sys.path.append('../keras')

import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, TimeDistributedMerge, TimeDistributedDense
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.layers.recurrent import LSTM


def load_data(data, dep_tree, maxlen, args):

    word2vecs = dep_tree.loadWord2VecMap()

    L = word2vecs[:args.wvecDim, :]

    Y = np.zeros((len(data), args.outputDim+1), dtype=np.float32)

    for i, (score, item) in enumerate(data):
        first_depTree, second_depTree = item

        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y[i, floor] = 1
        else:
            Y[i, floor] = ceil-sim 
            Y[i, ceil] = sim-floor

    Y = Y[:, 1:]

    X1 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)
    X2 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)

    scores = np.zeros((len(data)), dtype=np.float32)

    for i, (score, item) in enumerate(data):
        first_depTree, second_depTree = item

        for j, Node in enumerate(first_depTree.nodes):
            X1[i, j] =  L[:, Node.index]

        for k, Node in enumerate(second_depTree.nodes):
            X2[i, k] =  L[:, Node.index]

        scores[i] = score
        
    return X1, X2, Y, scores

def iterate_minibatches(inputs1, inputs2, targets, scores, batchsize, shuffle=False):
    assert len(inputs1) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs1[excerpt], inputs2[excerpt], targets[excerpt], scores[excerpt]

def build_network_2(args, maxlen=30):

    input_shape=(maxlen, args.wvecDim)

    print("Building model and compiling functions...")

    l_lstm_1 = keras.models.Sequential()
    l_lstm_1.add(keras.layers.recurrent.LSTM(output_dim=args.wvecDim, 
        return_sequences=True, input_shape=input_shape))
    l_lstm_1.add(keras.layers.core.Flatten())

    l_lstm_2 = keras.models.Sequential()
    l_lstm_2.add(keras.layers.recurrent.LSTM(output_dim=args.wvecDim, 
        return_sequences=True, input_shape=input_shape))
    l_lstm_2.add(keras.layers.core.Flatten())

    l_mul = keras.models.Sequential()
    l_mul.add(keras.layers.core.Merge([l_lstm_1, l_lstm_2], mode='mul'))
    l_mul.add(keras.layers.core.Dense(output_dim=args.wvecDim*10))

    l_sub = keras.models.Sequential()
    l_sub.add(keras.layers.core.Merge([l_lstm_1, l_lstm_2], mode='abs_sub'))
    l_sub.add(keras.layers.core.Dense(output_dim=args.wvecDim*10))

    model = keras.models.Sequential()
    model.add(keras.layers.core.Merge([l_mul, l_sub], mode='sum'))
    model.add(keras.layers.core.Reshape((10, args.wvecDim)))


    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    """
    model.add(keras.layers.convolutional.Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, maxlen, args.wvecDim)))

    model.add(keras.layers.core.Activation('relu'))
    model.add(keras.layers.convolutional.Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(keras.layers.core.Activation('relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(keras.layers.core.Dropout(0.25))
    """
    model.add(keras.layers.convolutional.Convolution1D(nb_filter=32, filter_length=2))
    model.add(keras.layers.core.Activation('relu'))
    model.add(keras.layers.core.Dropout(0.25))


    model.add(keras.layers.core.Flatten())
    model.add(keras.layers.core.Dense(50))
    model.add(keras.layers.core.Activation('relu'))
    model.add(keras.layers.core.Dropout(0.5))
    model.add(keras.layers.core.Dense(args.outputDim, init='uniform'))
    model.add(keras.layers.core.Activation('softmax'))

    #rms = RMSprop()
    #sgd = SGD(lr=0.1, decay=1e-6, mementum=0.9, nesterov=True)
    #adagrad = keras.optimizers.Adagrad(lr=0.05)
    adadelta = keras.optimizers.Adadelta(lr=2.0)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta)

    #train_fn = model.train_on_batch
    #test_fn = model.test_on_batch 
    #return train_fn, test_fn
    return model

def build_network_0(args, maxlen=30):

    print("Building model 0 and compiling functions...")

    """
    1. for each sentence, first do LSTM
    sent_1: (None, maxlen, wvecDim)
    sent_2: (None, maxlen, wvecDim)

    2. for each lstm output, do feature mean pooling(optional)
    sent_1: (None, maxlen, wvecDim/4)
    sent_2: (None, maxlen, wvecDim/4)

    2. Do multiply and abs_sub.
    mul: (None, maxlen, wvecDim/4)
    sub: (None, maxlen, wvecDim/4)

    3. hs= sigmoid(W1 * mul + W2* sub)

    4. pred = softmax(W*hs + b)

    """

    l_lstm_1 = Sequential()
    l_lstm_1.add(LSTM(output_dim=args.wvecDim, 
        return_sequences=True, input_shape=(maxlen, args.wvecDim)))
    l_lstm_1.add(Flatten())

    l_lstm_2 = Sequential()
    l_lstm_2.add(LSTM(output_dim=args.wvecDim, 
        return_sequences=True, input_shape=(maxlen, args.wvecDim)))
    l_lstm_2.add(Flatten())

    l_mul = Sequential()
    l_mul.add(Merge([l_lstm_1, l_lstm_2], mode='mul'))
    l_mul.add(Dense(output_dim=args.wvecDim))

    l_sub = Sequential()
    l_sub.add(Merge([l_lstm_1, l_lstm_2], mode='abs_sub'))
    l_sub.add(Dense(output_dim=args.wvecDim))

    model = Sequential()
    model.add(Merge([l_mul, l_sub], mode='sum'))
    model.add(Activation('sigmoid'))
    model.add(Dense(args.outputDim, init='uniform'))
    model.add(Activation('softmax'))

    #rms = RMSprop()
    #sgd = SGD(lr=0.1, decay=1e-6, mementum=0.9, nesterov=True)
    adagrad = Adagrad(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)

    return model


def build_network_1(args, maxlen=30):

    print("Building model 1 and compiling functions...")

    """
    1. for each sentence, first do LSTM
    sent_1: (None, maxlen, wvecDim)
    sent_2: (None, maxlen, wvecDim)

    2. Do multiply and abs_sub, and then mean pooling over maxlen.
    mul: (None, wvecDim)
    sub: (None, wvecDim)

    3. hs= sigmoid(W1 * mul + W2* sub)

    4. pred = softmax(W*hs + b)

    """
    l_lstm_1 = Sequential()
    l_lstm_1.add(LSTM(output_dim=args.wvecDim, return_sequences=True, input_shape=(maxlen, args.wvecDim)))

    l_lstm_2 = Sequential()
    l_lstm_2.add(LSTM(output_dim=args.wvecDim, return_sequences=True, input_shape=(maxlen, args.wvecDim)))


    l_mul = Sequential()
    l_mul.add(Merge([l_lstm_1, l_lstm_2], mode='mul'))
    l_mul.add(TimeDistributedMerge(mode='ave'))
    l_mul.add(Dense(args.hiddenDim, input_shape=(args.wvecDim,), init='uniform'))

    l_sub = Sequential()
    l_sub.add(Merge([l_lstm_1, l_lstm_2], mode='abs_sub'))
    l_sub.add(TimeDistributedMerge(mode='ave'))
    l_sub.add(Dense(args.hiddenDim, input_shape=(args.wvecDim,), init='uniform'))

    model = Sequential()
    model.add(Merge([l_mul, l_sub], mode='sum'))
    model.add(Activation('sigmoid'))
    model.add(Dense(args.outputDim, init='uniform'))
    model.add(Activation('softmax'))

    #rms = RMSprop()
    #sgd = SGD(lr=0.1, decay=1e-6, mementum=0.9, nesterov=True)
    adagrad = Adagrad(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)

    return model


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
    testTrees = tr.loadTrees("test")
    
    print "train number %d"%len(trainTrees)
    print "dev number %d"%len(devTrees)
    print "test number %d"%len(testTrees)

    maxlen = 36
    X1_train, X2_train, Y_train, scores_train = load_data(trainTrees, tr, maxlen, args)
    X1_dev, X2_dev, Y_dev, scores_dev = load_data(devTrees, tr, maxlen, args)
    X1_test, X2_test, Y_test, scores_test = load_data(testTrees, tr, maxlen, args)

    model = build_network_2(args, maxlen)

    print("Starting training...")
    best_dev_score = .0
    for epoch in range(args.epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X1_train, X2_train, Y_train, scores_train, args.minibatch, shuffle=True):
            inputs1, inputs2, targets, _ = batch
            err = model.train_on_batch([inputs1, inputs2], targets)
            train_batches += 1
            train_err += err

        # And a full pass over the validation data:
        
        val_err = 0
        val_batches = 0
        val_pearson = 0
        for batch in iterate_minibatches(X1_dev, X2_dev, Y_dev, scores_dev, 500, shuffle=False):
            inputs1, inputs2, targets, scores = batch
            preds = model.predict_proba([inputs1, inputs2])
            #val_err += 
            val_batches += 1

            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            val_pearson += pearson_score 


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


        print("  validation pearson:\t\t{:.2f} %".format(
            val_pearson / val_batches * 100))

    
