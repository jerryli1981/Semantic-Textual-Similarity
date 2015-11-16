import sys
import os
import time

import numpy as np
import theano

from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath('../keras'))

import keras

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

    
    Y_scores = np.zeros((len(data)), dtype=np.float32)
    labels = []

    for i, (label, score, l_tree, r_tree) in enumerate(data):

        for j, Node in enumerate(l_tree.nodes):
            X1[i, j] =  wordEmbeddings[:, Node.index]

        for k, Node in enumerate(r_tree.nodes):
            X2[i, k] =  wordEmbeddings[:, Node.index]

        labels.append(label)
        Y_scores[i] = score

    Y_labels = np.zeros((len(labels), args.numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.
        
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

def build_network(args,maxlen=36):

    input_shape=(maxlen, args.wvecDim)

    print("Building model and compiling functions...")

    l_lstm_1 = keras.models.Sequential()
    l_lstm_1.add(keras.layers.recurrent.LSTM(output_dim=args.wvecDim, 
        return_sequences=False, input_shape=input_shape))


    l_lstm_2 = keras.models.Sequential()
    l_lstm_2.add(keras.layers.recurrent.LSTM(output_dim=args.wvecDim, 
        return_sequences=False, input_shape=input_shape))


    l_mul = keras.models.Sequential()
    l_mul.add(keras.layers.core.Merge([l_lstm_1, l_lstm_2], mode='mul'))
    l_mul.add(keras.layers.core.Dense(output_dim=args.hiddenDim))

    l_sub = keras.models.Sequential()
    l_sub.add(keras.layers.core.Merge([l_lstm_1, l_lstm_2], mode='abs_sub'))
    l_sub.add(keras.layers.core.Dense(output_dim=args.hiddenDim))

    model = keras.models.Sequential()
    model.add(keras.layers.core.Merge([l_mul, l_sub], mode='sum'))
    model.add(keras.layers.core.Activation('sigmoid'))

    if args.task=="sts":
        model.add(keras.layers.core.Dense(args.rangeScores, init='uniform'))
    elif args.task == "ent":
        model.add(keras.layers.core.Dense(args.numLabels, init='uniform'))

    model.add(keras.layers.core.Activation('softmax'))


    if args.optimizer == "sgd":
        optimizer = keras.optimizers.SGD(lr=args.step, decay=1e-6, mementum=0.9, nesterov=True)
    elif args.optimizer == "adagrad":
        optimizer = keras.optimizers.Adagrad(args.step)
    elif args.optimizer == "adadelta":
        optimizer = keras.optimizers.Adadelta(lr=args.step)
    elif args.optimizer == "rms":
        optimizer = keras.optimizers.RMSprop(lr=args.step)
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

    wordEmbeddings = tr.loadWord2VecMap()[:args.wvecDim, :]
    
    X1_train, X2_train, Y_labels_train, Y_scores_train, Y_scores_pred_train = load_data(trainTrees, wordEmbeddings, args)
    X1_dev, X2_dev, Y_labels_dev, Y_scores_dev, Y_scores_pred_dev = load_data(devTrees, wordEmbeddings, args)
    X1_test, X2_test, Y_labels_test, Y_scores_test, Y_scores_pred_test = load_data(testTrees, wordEmbeddings, args)

    train_fn, val_fn, predict_proba= build_network(args)

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