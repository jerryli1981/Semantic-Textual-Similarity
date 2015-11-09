import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.recurrent import LSTM

from scipy.stats import pearsonr


def load_data(data, dep_tree, args):

    word2vecs = dep_tree.loadWord2VecMap()

    L = word2vecs[:args.wvecDim, :]

    maxlen_1 = 0
    maxlen_2 = 0
    Y = np.zeros((len(data), args.outputDim+1), dtype=theano.config.floatX)

    for i, (score, item) in enumerate(data):
        first_depTree, second_depTree = item
        first_len = len(first_depTree.nodes)
        second_len = len(second_depTree.nodes)
        if first_len > maxlen_1:
            maxlen_1 = first_len
        if second_len > maxlen_2:
            maxlen_2 = second_len

        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y[i, floor] = 1
        else:
            Y[i, floor] = ceil-sim 
            Y[i, ceil] = sim-floor

    Y = Y[:, 1:]

    X1 = np.zeros((len(data), maxlen_1, args.wvecDim), dtype=theano.config.floatX)
    X2 = np.zeros((len(data), maxlen_2, args.wvecDim), dtype=theano.config.floatX)

    for i, (score, item) in enumerate(data):
        first_depTree, second_depTree = item

        for j, Node in enumerate(first_depTree.nodes):
            X1[i, j] =  L[:, Node.index]

        for k, Node in enumerate(second_depTree.nodes):
            X2[i, k] =  L[:, Node.index]

    return X1, X2, Y, maxlen_1, maxlen_2

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
    
    X1_train, X2_train, Y_train, maxlen_1, maxlen_2 = load_data(trainTrees, tr, args)
    X1_dev, X2_dev, Y_dev, _,_ = load_data(devTrees, tr, args)
    X1_test, X2_test, Y_test,_,_ = load_data(testTrees, tr, args)

    # build the model: 2 stacked LSTM
    print('Build model...')
    first_rep_layer = Sequential()
    first_rep_layer.add(LSTM(args.wvecDim, return_sequences=False, input_shape=(maxlen_1, args.wvecDim)))
    #first_model.add(Dropout(0.2))
    #first_model.add(LSTM(args.wvecDim, return_sequences=False))
    #first_model.add(Dropout(0.2))
    #first_model.add(Dense(args.wvecDim, input_shape=(args.wvecDim,)))

    second_rep_layer = Sequential()
    second_rep_layer.add(LSTM(args.wvecDim, return_sequences=False, input_shape=(maxlen_2, args.wvecDim)))
    #second_model.add(Dropout(0.2))
    #second_model.add(LSTM(args.wvecDim, return_sequences=False))
    #second_model.add(Dropout(0.2))
    #second_model.add(Dense(args.wvecDim, input_shape=(args.wvecDim,)))

    mul_layer = Sequential()
    mul_layer.add(Merge([first_rep_layer, second_rep_layer], mode='mul'))
    mul_layer.add(Dense(args.hiddenDim, input_shape=(args.wvecDim,)))

    sub_layer = Sequential()
    sub_layer.add(Merge([first_rep_layer, second_rep_layer], mode='abs_sub'))
    sub_layer.add(Dense(args.hiddenDim, input_shape=(args.wvecDim,)))

    model = Sequential()
    model.add(Merge([mul_layer, sub_layer], mode='sum'))
    model.add(Dense(args.hiddenDim, input_shape=(args.wvecDim,)))
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.2))
    #model.add(Dense(50))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    rms = RMSprop()
    sgd = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print "begin to train model"
    corrects = [] 
    for i, (score, item) in enumerate(devTrees):
        corrects.append(score)
    #model.fit([X1_train, X2_train], Y_train, batch_size=args.minibatch, nb_epoch=args.epochs, show_accuracy=True, verbose=2, validation_data=([X1_dev, X2_dev],  Y_dev))

    model.fit([X1_train, X2_train], Y_train, batch_size=args.minibatch, nb_epoch=100)
    
    epoch = 0
    best_dev_score  = 0.
    while (epoch < args.epochs):
        epoch = epoch + 1 

        batches = [trainTrees[idx : idx + args.minibatch] for idx in xrange(0, len(trainTrees), args.minibatch)]

        for index, batchData in enumerate(batches):
            X1_train_batch, X2_train_batch, Y_train_batch, _,_= load_data(batchData, tr, args)
            model.train_on_batch([X1_train_batch, X2_train_batch], Y_train_batch)

        preds = model.predict([X1_dev, X2_dev])
        predictScores = preds.dot(np.array([1,2,3,4,5]))
        guesses = predictScores.tolist()
        dev_score = pearsonr(corrects,guesses)[0]
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print "iter:%d dev_score: %f best_dev_score %f"%(epoch, dev_score, best_dev_score)
        else:
            print "iter:%d dev_score: %f"%(epoch, dev_score)

    preds = model.predict([X1_dev, X2_dev])
    predictScores = preds.dot(np.array([1,2,3,4,5]))
    guesses = predictScores.tolist()
    print pearsonr(corrects,guesses)[0]
    