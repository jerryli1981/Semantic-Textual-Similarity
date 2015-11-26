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

from utils import loadWord2VecMap, iterate_minibatches, read_sequence_dataset, read_sequence_dataset_embedding


def build_network_graph_embedding(args, wordEmbeddings, maxlen=36, reg=0.5*1e-4):
 
    print("Building graph model with embeddings...")
    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    batch_size = args.minibatch

    model = Graph()
    model.add_input(name='input1', input_shape=(maxlen,wordDim), dtype='float')
    model.add_input(name='input2', input_shape=(maxlen,wordDim), dtype='float')
    
    lstm_1 = LSTM(output_dim=args.lstmDim, return_sequences=False, 
        input_shape=(maxlen, wordDim))

    lstm_1.regularizers = [l2(reg)] * 12
    for i in range(12):    
        lstm_1.regularizers[i].set_param(lstm_1.get_params()[0][i])

    model.add_node(lstm_1, input='input1', name='lstm1')


    lstm_2 = LSTM(output_dim=args.lstmDim, return_sequences=False, 
        input_shape=(maxlen, wordDim))
    
    model.add_node(lstm_2, input='input2', name='lstm2')

    model.add_node(Activation('linear'), inputs=['lstm1', 'lstm2'], name='mul_merge', merge_mode='mul')
    model.add_node(Activation('linear'), inputs=['lstm1', 'lstm2'], name='abs_merge', merge_mode='abs_sub')

    d = Dense(output_dim=args.hiddenDim, W_regularizer=l2(reg),b_regularizer=l2(reg))
    model.add_node(d, inputs=['mul_merge', 'abs_merge'], name="concat", merge_mode='concat')
    model.add_node(Activation('sigmoid'), input='concat', name='sig')

    if args.task=="sts":
        model.add_node(Dense(5,W_regularizer=l2(reg), b_regularizer=l2(reg)), input='sig', name='den')
    elif args.task == "ent":
        model.add_node(Dense(3,W_regularizer=l2(reg), b_regularizer=l2(reg)), input='sig', name='den')

    model.add_node(Activation('softmax'), input='den', name='softmax')
    model.add_output(name='softmax_out', input='softmax')

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

    model.compile(optimizer=optimizer, loss={'softmax_out':'kl_divergence'})

    train_fn = model.train_on_batch
    val_fn = model.test_on_batch 
    predict_proba = model.predict

    return train_fn, val_fn, predict_proba


def build_network_graph_index(args, wordEmbeddings, maxlen=36, reg=0.5*1e-4):
 
    print("Building graph model with indexes...")
    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    batch_size = args.minibatch

    model = Graph()
    model.add_input(name='input1', input_shape=(maxlen,), dtype='int')
    model.add_input(name='input2', input_shape=(maxlen,), dtype='int')

    model.add_node(Embedding(input_dim=vocab_size, output_dim=wordDim, 
        mask_zero=True, weights=[wordEmbeddings.T],input_length=maxlen), input='input1', name='emb1')

    lstm_1 = LSTM(output_dim=args.lstmDim, return_sequences=False, 
        input_shape=(maxlen, wordDim))

    lstm_1.regularizers = [l2(reg)] * 12
    for i in range(12):    
        lstm_1.regularizers[i].set_param(lstm_1.get_params()[0][i])

    model.add_node(lstm_1, input='emb1', name='lstm1')

    model.add_node(Embedding(input_dim=vocab_size, output_dim=wordDim, 
        mask_zero=True, weights=[wordEmbeddings.T],input_length=maxlen), input='input2', name='emb2')

    lstm_2 = LSTM(output_dim=args.lstmDim, return_sequences=False, 
        input_shape=(maxlen, wordDim))
    
    model.add_node(lstm_2, input='emb2', name='lstm2')

    model.add_node(Activation('linear'), inputs=['lstm1', 'lstm2'], name='mul_merge', merge_mode='mul')
    model.add_node(Activation('linear'), inputs=['lstm1', 'lstm2'], name='abs_merge', merge_mode='abs_sub')

    d = Dense(output_dim=args.hiddenDim, W_regularizer=l2(reg),b_regularizer=l2(reg))
    model.add_node(d, inputs=['mul_merge', 'abs_merge'], name="concat", merge_mode='concat')
    model.add_node(Activation('sigmoid'), input='concat', name='sig')

    if args.task=="sts":
        model.add_node(Dense(5,W_regularizer=l2(reg), b_regularizer=l2(reg)), input='sig', name='den')
    elif args.task == "ent":
        model.add_node(Dense(3,W_regularizer=l2(reg), b_regularizer=l2(reg)), input='sig', name='den')

    model.add_node(Activation('softmax'), input='den', name='softmax')
    model.add_output(name='softmax_out', input='softmax')

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

    model.compile(optimizer=optimizer, loss={'softmax_out':'kl_divergence'})

    train_fn = model.train_on_batch
    val_fn = model.test_on_batch 
    predict_proba = model.predict

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

    # Load the dataset
    print("Loading data...")
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    
    X1_train, X1_mask_train, X2_train, X2_mask_train, Y_labels_train, Y_scores_train, Y_scores_pred_train = \
        read_sequence_dataset_embedding(sick_dir, "train", wordEmbeddings)
    X1_dev, X1_mask_dev, X2_dev, X2_mask_dev, Y_labels_dev, Y_scores_dev, Y_scores_pred_dev = \
        read_sequence_dataset_embedding(sick_dir, "dev", wordEmbeddings)
    X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, Y_scores_test, Y_scores_pred_test = \
        read_sequence_dataset_embedding(sick_dir, "test", wordEmbeddings)

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings.astype(np.float32)

    train_fn, val_fn, predict_proba= build_network_graph_embedding(args, wordEmbeddings)

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
                train_err += train_fn({"input1":inputs1, "input2":inputs2, "softmax_out":scores_pred})
            elif args.task == "ent":
                train_err += train_fn({"input1":inputs1, "input2":inputs2, "softmax_out":labels})
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
                err = val_fn({"input1":inputs1, "input2":inputs2, "softmax_out":scores_pred})
                preds = predict_proba({"input1":inputs1, "input2":inputs2})
                preds = preds['softmax_out']
                predictScores = preds.dot(np.array([1,2,3,4,5]))
                guesses = predictScores.tolist()
                scores = scores.tolist()
                pearson_score = pearsonr(scores,guesses)[0]
                val_pearson += pearson_score 

            elif args.task == "ent":
                err, acc = val_fn({"input1":inputs1, "input2":inputs2, "softmax_out":labels}, accuracy=True)
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

    test_err = 0
    test_acc = 0
    test_pearson = 0
    test_batches = 0
    for batch in iterate_minibatches(X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, 
        Y_scores_test, Y_scores_pred_test, len(X1_test), shuffle=False):

        inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

        if args.task == "sts":
            err = val_fn({"input1":inputs1, "input2":inputs2, "softmax_out":scores_pred})
            preds = predict_proba({"input1":inputs1, "input2":inputs2})
            preds = preds['softmax_out']
            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            test_pearson += pearson_score 

        elif args.task == "ent":
            err, acc = val_fn({"input1":inputs1, "input2":inputs2, "softmax_out":labels}, accuracy=True)
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