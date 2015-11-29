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
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2,activity_l2

from utils import loadWord2VecMap, iterate_minibatches, read_sequence_dataset

def build_network(args, wordEmbeddings, maxlen=36, reg=0.5*1e-4):
 
    print("Building sequential model ...")
    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]

    l_lstm_1 = Sequential()
    """
    M_1 = Masking(mask_value=vocab_size-1)
    M_1._input_shape = (maxlen, wordDim)
    l_lstm_1.add(M_1)
    """
    l_lstm_1.add(Embedding(input_dim=vocab_size, output_dim=wordDim, 
        mask_zero=True, weights=[wordEmbeddings.T],input_length=maxlen))

    #l_lstm_1.add(Embedding(input_dim=n_symbols, output_dim=300, input_length=maxlen))

    l_lstm_1.add(LSTM(output_dim=args.lstmDim, return_sequences=False, 
        input_shape=(maxlen, wordDim)))
    l_lstm_1.add(Dropout(0.1))

    l_lstm_1.layers[1].regularizers = [l2(reg)] * 12
    for i in range(12):    
        l_lstm_1.layers[1].regularizers[i].set_param(l_lstm_1.layers[1].get_params()[0][i])

    l_lstm_2 = Sequential()

    """
    M_2 = Masking(mask_value=vocab_size-1)
    M_2._input_shape = (maxlen, wordDim)
    l_lstm_2.add(M_2)
    """
    
    l_lstm_2.add(Embedding(input_dim=vocab_size, output_dim=wordDim, 
        mask_zero=True, weights=[wordEmbeddings.T],input_length=maxlen))
    
    #l_lstm_2.add(Embedding(input_dim=n_symbols, output_dim=300, input_length=maxlen))

    l_lstm_2.add(LSTM(output_dim=args.lstmDim, return_sequences=False, 
        input_shape=(maxlen, wordDim)))
    l_lstm_2.add(Dropout(0.1))
    
    l_lstm_2.layers[1].regularizers = [l2(reg)] * 12
    for i in range(12):    
        l_lstm_2.layers[1].regularizers[i].set_param(l_lstm_2.layers[1].get_params()[0][i])
    

    l_mul = Sequential()
    l_mul.add(Merge([l_lstm_1, l_lstm_2], mode='mul'))
    #l_mul.add(Dense(output_dim=150,W_regularizer=l2(reg),b_regularizer=l2(reg)))

    l_sub = Sequential()
    l_sub.add(Merge([l_lstm_1, l_lstm_2], mode='abs_sub'))
    #l_sub.add(Dense(output_dim=150,W_regularizer=l2(reg),b_regularizer=l2(reg)))

    model = Sequential()
    model.add(Merge([l_mul, l_sub], mode='concat', concat_axis=-1))
    model.add(Dense(output_dim=args.hiddenDim,W_regularizer=l2(reg),b_regularizer=l2(reg)))
    #model.add(Merge([l_mul,l_sub], mode='sum'))

    model.add(Activation('sigmoid'))

    if args.task=="sts":
        model.add(Dense(5,W_regularizer=l2(reg), b_regularizer=l2(reg)))
    elif args.task == "ent":
        model.add(Dense(3,W_regularizer=l2(reg), b_regularizer=l2(reg)))

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
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=20)
    parser.add_argument("--step",dest="step",type=float,default=0.01)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--lstmDim",dest="lstmDim",type=int,default=30)
    parser.add_argument("--task",dest="task",type=str,default=None)
    args = parser.parse_args()

    print("Loading data...")
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    
    X1_train, X1_mask_train, X2_train, X2_mask_train, Y_labels_train, Y_scores_train, Y_scores_pred_train = \
        read_sequence_dataset(sick_dir, "train")
    X1_dev, X1_mask_dev, X2_dev, X2_mask_dev, Y_labels_dev, Y_scores_dev, Y_scores_pred_dev = \
        read_sequence_dataset(sick_dir, "dev")
    X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, Y_scores_test, Y_scores_pred_test = \
        read_sequence_dataset(sick_dir, "test")

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings.astype(np.float32)

    train_fn, val_fn, predict_proba= build_network(args, wordEmbeddings)

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

        for batch in iterate_minibatches(X1_dev, X1_mask_dev, X2_dev, X2_mask_dev, Y_labels_dev, Y_scores_dev, 
            Y_scores_pred_dev, len(X1_dev), shuffle=False):

            inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

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
    for batch in iterate_minibatches(X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, 
        Y_scores_test, Y_scores_pred_test, len(X1_test), shuffle=False):

        inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

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