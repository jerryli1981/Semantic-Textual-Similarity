import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import random

from scipy.stats import pearsonr

class rnn_mlp_model(object):

    def __init__(self, rng, n_rel, wvecDim, hiddenDim, outputDim, L):

        self.wvecDim = wvecDim
        self.outputDim = outputDim

        ##########################################
        # below is Recursive NN params
        #
        ##########################################

        self.L = theano.shared(value=L, name='L',
                          borrow=True)

        # Relation layer parameters
        self.WR = theano.shared(
                value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (wvecDim + wvecDim)),
                        high=np.sqrt(6. / (wvecDim + wvecDim)),
                        size=(n_rel, wvecDim, wvecDim)
                    ), dtype=theano.config.floatX), 
                name='WR',
                borrow=True)

        self.WV = theano.shared(
                value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (wvecDim + wvecDim)),
                        high=np.sqrt(6. / (wvecDim + wvecDim)),
                        size=(wvecDim, wvecDim)
                    ),dtype=theano.config.floatX),
                name='WV',
                borrow=True)

        self.b_rnn = theano.shared(
                value=np.zeros((wvecDim,), dtype=theano.config.floatX), 
                name='b_rnn',
                borrow=True)

        ##########################################
        # below is mlp params 
        #
        ##########################################

        self.W_1 = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (wvecDim + hiddenDim)),
                    high=np.sqrt(6. / (wvecDim + hiddenDim)),
                    size=(wvecDim, hiddenDim)
                ),dtype=theano.config.floatX), 
            name='W_1', 
            borrow=True)


        self.W_2 = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (wvecDim + hiddenDim)),
                    high=np.sqrt(6. / (wvecDim + hiddenDim)),
                    size=(wvecDim, hiddenDim)
            ),dtype=theano.config.floatX), 
            name='W_2', 
            borrow=True)


        self.b_mlp_hidden = theano.shared(
            value=np.zeros((hiddenDim,), dtype=theano.config.floatX), 
            name='b_mlp_hidden', 
            borrow=True)

        self.W_out_mlp = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (hiddenDim + outputDim)),
                    high=np.sqrt(6. / (hiddenDim + outputDim)),
                    size=(hiddenDim, outputDim)
                ),
                dtype=theano.config.floatX
            ), 
            name='W_out_mlp', 
            borrow=True)


        # initialize the baises b as a vector of n_out 0s
        self.b_out_mlp = theano.shared(
            value=np.zeros((outputDim,),dtype=theano.config.floatX),
            name='b_out_mlp',
            borrow=True
        )


        self.L2_sqr_rnn = (
            (self.L ** 2).sum() + (self.WR ** 2).sum()
            + (self.WV ** 2).sum() + (self.b_rnn ** 2).sum()
        )

        self.L2_sqr_mlp = (
            (self.W_1 ** 2).sum() + (self.W_2 ** 2).sum() + (self.b_mlp_hidden **2).sum()
            + (self.W_out_mlp ** 2).sum() + (self.b_out_mlp ** 2).sum()
        )

        self.lambda_const = 0.01 
        self.param_error = self.lambda_const * (self.L2_sqr_rnn + self.L2_sqr_mlp)


        inputs_1 = T.fmatrix('inputs_1')  # n * d, the data is presented as one sentence output
        inputs_2 = T.fmatrix('inputs_2')  # n * d, the data is presented as one sentence output
        targets = T.fmatrix('targets')  # n * d, the target distribution\

        hidden_output = T.tanh(T.dot(inputs_1, self.W_1) + T.dot(inputs_2, self.W_2) + self.b_mlp_hidden)

        p_y_given_x = T.nnet.softmax(T.dot(hidden_output, self.W_out_mlp) + self.b_out_mlp)

        cross_entropy = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, targets))

        self.cost_function = theano.function([inputs_1 , inputs_2 ,targets], cross_entropy, allow_input_downcast=True)

        self.val_fn = theano.function([inputs_1, inputs_2, targets], [cross_entropy, p_y_given_x], allow_input_downcast=True)

        self.stack = [self.L, self.WR, self.WV, self.b_rnn, 
                        self.W_1, self.W_2, self.b_mlp_hidden, self.W_out_mlp, self.b_out_mlp]

    def step(self, tree):
        #because many training epoch. 
        tree.resetFinished()

        to_do = []
        to_do.append(tree.root)

        while to_do:
        
            curr = to_do.pop(0)
            curr.vec = self.L.get_value()[:, curr.index]

            # node is leaf
            if len(curr.kids) == 0:

                curr.hAct = np.tanh(np.dot(curr.vec, self.WV.get_value()) + self.b_rnn.get_value())
                curr.finished=True

            else:

                #check if all kids are finished
                all_done = True
                for index, rel in curr.kids:
                    node = tree.nodes[index]
                    if not node.finished:
                        to_do.append(node)
                        all_done = False

                if all_done:

                    sum = np.zeros(self.wvecDim)
                    for i, rel in curr.kids:
                        W_rel = self.WR.get_value()[rel.index] # d * d
                        sum += np.dot(tree.nodes[i].hAct, W_rel) 

                    curr.hAct = np.tanh(sum + np.dot(curr.vec, self.WV.get_value()) + self.b_rnn.get_value())
        
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        return tree.root.hAct


    def forwardProp(self, inputs1, inputs2, targets):

        cost = 0.0

        for l_tree, r_tree, y in zip(inputs1, inputs2, targets):
            
            l_rep = self.step(l_tree)
            r_rep = self.step(r_tree)

            l_rep = l_rep.reshape((1, self.wvecDim))
            r_rep = r_rep.reshape((1, self.wvecDim))

            y = y.reshape((1, self.outputDim))

            cost += self.cost_function(l_rep, r_rep, y) + self.param_error

        return cost

    def predict(self, inputs1, inputs2, targets):

        l_reps = np.zeros((len(inputs1), self.wvecDim))
        r_reps = np.zeros((len(inputs1), self.wvecDim))

        for i, (l_tree, r_tree) in enumerate(zip(inputs1, inputs2)):
            
            l_reps[i, :] = self.step(l_tree)
            r_reps[i, :] = self.step(r_tree)

        err, pred = self.val_fn(l_reps,r_reps, targets)

        return err, pred


def load_data(data, dep_tree, args):

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

    X1 = []
    X2 = []

    scores = np.zeros((len(data)), dtype=np.float32)

    for i, (score, item) in enumerate(data):
        first_depTree, second_depTree = item
        X1.append(first_depTree)
        X2.append(second_depTree)
        scores[i] = score
        
    return X1, X2, Y, scores

def iterate_minibatches(inputs1, inputs2, targets, scores, batchsize, shuffle=False):

    assert len(inputs1) == len(targets)

    if shuffle:
        indices = [ i for i in range(len(inputs1))]
        random.shuffle(indices)

        ninputs1 = [inputs1[i] for i in indices]
        batches_inputs1 = [ninputs1[idx : idx + batchsize] for idx in xrange(0, len(inputs1), batchsize)]

        ninputs2 = [inputs2[i] for i in indices]
        batches_inputs2 = [ninputs2[idx : idx + batchsize] for idx in xrange(0, len(inputs2), batchsize)]

        ntargets = [targets[i] for i in indices]
        batches_targets = [ntargets[idx : idx + batchsize] for idx in xrange(0, len(targets), batchsize)]

        nscores = [scores[i] for i in indices]
        batches_scores = [nscores[idx : idx + batchsize] for idx in xrange(0, len(scores), batchsize)]

        return zip(batches_inputs1, batches_inputs2, batches_targets,batches_scores)
    else:

        indices = [ i for i in range(len(inputs1))]
        ninputs1 = [inputs1[i] for i in indices]
        batches_inputs1 = [ninputs1[idx : idx + batchsize] for idx in xrange(0, len(inputs1), batchsize)]

        ninputs2 = [inputs2[i] for i in indices]
        batches_inputs2 = [ninputs2[idx : idx + batchsize] for idx in xrange(0, len(inputs2), batchsize)]

        ntargets = [targets[i] for i in indices]
        batches_targets = [ntargets[idx : idx + batchsize] for idx in xrange(0, len(targets), batchsize)]

        nscores = [scores[i] for i in indices]
        batches_scores = [nscores[idx : idx + batchsize] for idx in xrange(0, len(scores), batchsize)]

        return zip(batches_inputs1, batches_inputs2, batches_targets,batches_scores)


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

    # Load the dataset
    print("Loading data...")
    import dependency_tree as tr     
    trainTrees = tr.loadTrees("train")
    #trainTrees = trainTrees[:100]
    devTrees = tr.loadTrees("dev")
    #devTrees = devTrees[:10]
    testTrees = tr.loadTrees("test")
    
    print "train number %d"%len(trainTrees)
    print "dev number %d"%len(devTrees)
    print "test number %d"%len(testTrees)
    
    X1_train, X2_train, Y_train, scores_train = load_data(trainTrees, tr, args)
    X1_dev, X2_dev, Y_dev, scores_dev = load_data(devTrees, tr, args)
    X1_test, X2_test, Y_test, scores_test = load_data(testTrees, tr, args)


    rng = np.random.RandomState(1234)
    word2vecs=tr.loadWord2VecMap()
    L = word2vecs[:args.wvecDim, :]
    n_rel=len(tr.loadRelMap())

    print("Building model...")
    model = rnn_mlp_model(rng, n_rel, args.wvecDim, args.hiddenDim, args.outputDim, L)

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(args.epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X1_train, X2_train, Y_train, scores_train, args.minibatch, shuffle=True):
            inputs1, inputs2, targets, _ = batch
            cost = model.forwardProp(inputs1, inputs2, targets)
            gparams = [T.grad(cost, param) for param in model.stack]
            updates = [(param, param - args.step * gtheta) for param, gtheta in zip(model.stack, gparams)]

            for e in updates:
                tmp_new = e[1].eval({})
                e[0].set_value(tmp_new)

            train_err += cost.eval({})

            train_batches += 1
            cost = 0.0

        # And a full pass over the validation data:
        
        
        val_err = 0
        val_batches = 0
        val_pearson = 0
        for batch in iterate_minibatches(X1_dev, X2_dev, Y_dev, scores_dev, 500, shuffle=False):
            inputs1, inputs2, targets, scores = batch
            err, preds = model.predict(inputs1, inputs2, targets)
            val_err += err
            val_batches += 1
            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            #scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            val_pearson += pearson_score 

        
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


        print("  validation pearson:\t\t{:.2f} %".format(
            val_pearson / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_pearson = 0
    test_batches = 0
    for batch in iterate_minibatches(X1_test, X2_test, Y_test, scores_test, 4500, shuffle=False):
        inputs1, inputs2, targets, scores = batch
        err, preds = model.predict(inputs1, inputs2, targets)
        test_err += err
        test_batches += 1

        predictScores = preds.dot(np.array([1,2,3,4,5]))
        guesses = predictScores.tolist()
        #scores = scores.tolist()
        pearson_score = pearsonr(scores,guesses)[0]
        test_pearson += pearson_score 


    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test pearson:\t\t{:.2f} %".format(
        test_pearson / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    
