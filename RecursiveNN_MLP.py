import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import random

from scipy.stats import pearsonr

from collections import OrderedDict

class rnn_mlp_model(object):

    def __init__(self, rng, n_rel, wvecDim, hiddenDim, outputDim, L, input_shape):

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

        self.lambda_const = 0.0001
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

        (mini_batch, seq_len, n_children, n_ele) = input_shape

        self.mb = mini_batch
        self.seq_len = seq_len
        self.n_children = n_children
        self.n_ele = n_ele

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


    def get_output_for(self, inputs):

        reps = np.zeros((len(inputs), self.wvecDim))

        for i, input_tr in enumerate(inputs):
            
            #tree.resetFinished() -2: unfinished, -3:finished
            for node_idx in range(self.seq_len):
                node = input_tr[node_idx]
                node[0,4] = -2

            #input_tr is tensor3 (seq_len, n_children, n_ele)
            root_node = input_tr[-1]
            
            to_do = []
            to_do.append(root_node)

            while to_do:
     
                curr_node = to_do.pop(0)
                curr_vec = self.L.get_value()[:, curr_node[0,0]]

                #if node is leaf, len(curr.kids) == 0:
                if curr_node[0,1] == -1:

                    curr_hAct = np.tanh(np.dot(curr_vec, self.WV.get_value()) + self.b_rnn.get_value())
                    curr_node[0,5:] = curr_hAct

                    #curr.finished = True
                    curr_node[0, 4] = -3

                else:
                    #check if all kids are finished
                    all_done = True

                    for kid_idx in range(self.n_children):

                        if curr_node[kid_idx, 0] == -1:
                            continue

                        kid_local_idx = curr_node[kid_idx, 1] 

                        kid_node = input_tr[kid_local_idx]

                        if kid_node[0, 4] == -2:

                            to_do.append(kid_node)
                            all_done = False

                    if all_done:

                        sum_ = np.zeros(self.wvecDim)
                        for kid_idx in range(self.n_children):

                            if curr_node[kid_idx, 0] == -1:
                                continue

                            rel_idx = curr_node[kid_idx, 3]

                            W_rel = self.WR.get_value()[rel_idx]

                            kid_local_idx = curr_node[kid_idx, 1] 

                            kid_node = input_tr[kid_local_idx]

                            kid_node_hAct = kid_node[0,5:]

                            sum_ += np.dot(kid_node_hAct, W_rel)
                    
                        curr_hAct = np.tanh(sum_ + np.dot(curr_vec, self.WV.get_value()) + self.b_rnn.get_value())
                        curr_node[0,5:] = curr_hAct

                        #curr.finished = True
                        curr_node[0, 4] = -3

                    else:
                        to_do.append(curr_node)
            
            reps[i] = root_node[0,5:]
        
        return reps


    def forwardProp(self, inputs1, inputs2, targets):

        l_reps = self.get_output_for(inputs1)
        r_reps = self.get_output_for(inputs2)

        return self.cost_function(l_reps, r_reps, targets) + self.param_error

    def predict(self, inputs1, inputs2, targets):

        l_reps = self.get_output_for(inputs1)
        r_reps = self.get_output_for(inputs2)

        err, pred = self.val_fn(l_reps,r_reps, targets)

        return err, pred

def load_data(data, args, seq_len=37, n_children=6, n_ele=5, unfinished_flag=-2):

    Y = np.zeros((len(data), args.outputDim+1), dtype=np.float32)
    scores = np.zeros((len(data)), dtype=np.float32)

    # to store hidden representation
    storage_dim = n_ele + args.wvecDim

    X1 = np.zeros((len(data), seq_len, n_children, storage_dim), dtype=np.float32)
    X1.fill(-1.0)
    X2 = np.zeros((len(data), seq_len, n_children, storage_dim), dtype=np.float32)
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
            if len(Node.kids) != 0:
                children = np.zeros((n_children, storage_dim), dtype=np.float32)
                children.fill(-1.0)
                for m, (depIdx, rel) in enumerate(Node.kids):
                    depGlobalIdx = first_t.nodes[depIdx].index
                    children[m,0] = Node.index
                    children[m,1] = depIdx
                    children[m,2] = depGlobalIdx
                    children[m,3] = rel.index
                    children[m,4] = unfinished_flag
                X1[i, j] = children
            else:
                children = np.zeros((n_children, storage_dim), dtype=np.float32)
                children.fill(-1.0)
                #children[0] = np.array([Node.index, -1, -1, -1, unfinished_flag], dtype=np.int32)
                children[0,0] = Node.index
                children[0,1] = -1
                children[0,2] = -1
                children[0,3] = -1
                children[0,4] = unfinished_flag
                X1[i, j] = children
        
        first_root = np.zeros((n_children, storage_dim), dtype=np.float32)
        first_root.fill(-1.0)
        for c, (depIdx, rel) in enumerate(first_t.root.kids):
            depGlobalIdx = first_t.nodes[depIdx].index
            #first_root[c] = np.array([first_t.root.index, depIdx, depGlobalIdx, rel.index, unfinished_flag], dtype=np.int32)
            first_root[c,0] = first_t.root.index
            first_root[c,1] = depIdx
            first_root[c,2] = depGlobalIdx
            first_root[c,3] = rel.index
            first_root[c,4] = unfinished_flag
        X1[i, -1] = first_root

        s_idxSet = set()
        for govIdx, depIdx in second_t.dependencies:
            s_idxSet.add(govIdx)
            s_idxSet.add(depIdx)
        
        for j, Node in enumerate(second_t.nodes):
            if j not in s_idxSet:
                continue
            if len(Node.kids) != 0:
                children = np.zeros((n_children, storage_dim), dtype=np.float32)
                children.fill(-1.0)
                for m, (depIdx, rel) in enumerate(Node.kids):
                    depGlobalIdx = second_t.nodes[depIdx].index
                    children[m,0] = Node.index
                    children[m,1] = depIdx
                    children[m,2] = depGlobalIdx
                    children[m,3] = rel.index
                    children[m,4] = unfinished_flag
                    #children[m] = np.array([Node.index, depIdx, depGlobalIdx, rel.index, unfinished_flag], dtype=np.int32)
                X2[i, j] = children

            else:
                children = np.zeros((n_children, storage_dim), dtype=np.float32)
                children.fill(-1.0)
                #children[0] = np.array([Node.index, -1, -1, -1, unfinished_flag], dtype=np.int32)
                children[0,0] = Node.index
                children[0,1] = -1
                children[0,2] = -1
                children[0,3] = -1
                children[0,4] = unfinished_flag
                X2[i, j] = children

        second_root = np.zeros((n_children, storage_dim), dtype=np.float32)
        second_root.fill(-1.0)
        for c, (depIdx, rel) in enumerate(second_t.root.kids):
            depGlobalIdx = second_t.nodes[depIdx].index
            #second_root[c] = np.array([second_t.root.index, depIdx, depGlobalIdx, rel.index, unfinished_flag], dtype=np.int32)
            second_root[c,0] = second_t.root.index
            second_root[c,1] = depIdx
            second_root[c,2] = depGlobalIdx
            second_root[c,3] = rel.index
            second_root[c,4] = unfinished_flag

        X2[i, -1] = second_root
        
        scores[i] = score

    Y = Y[:, 1:]

    input_shape = (len(data), seq_len, n_children, n_ele)
      
    return X1, X2, Y, scores, input_shape

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

def sgd_updates_adagrad(params,cost):
    
    updates = OrderedDict({})
    
    exp_sqr_grads = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
        exp_sqr_grads[param] = theano.shared(value=empty,name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        gparams.append(gp)

    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        up_exp_sg = exp_sg + T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  gp * ( 1./T.sqrt(up_exp_sg) )
        stepped_param = param - step
        updates[param] = stepped_param 

    ret = []
    for key, value in updates.iteritems(): 
        ret.append((key,value))    
    return ret 


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
    devTrees = tr.loadTrees("dev")
    testTrees = tr.loadTrees("test")
    
    print "train number %d"%len(trainTrees)
    print "dev number %d"%len(devTrees)
    print "test number %d"%len(testTrees)
    
    X1_train, X2_train, Y_train, scores_train,input_shape = load_data(trainTrees, args)
    X1_dev, X2_dev, Y_dev, scores_dev,_ = load_data(devTrees, args)
    X1_test, X2_test, Y_test, scores_test,_ = load_data(testTrees, args)


    rng = np.random.RandomState(1234)
    word2vecs=tr.loadWord2VecMap()
    L = word2vecs[:args.wvecDim, :]
    n_rel=len(tr.loadRelMap())

    print("Building model...")
    model = rnn_mlp_model(rng, n_rel, args.wvecDim, args.hiddenDim, args.outputDim, L, input_shape)

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
            #gparams = [T.grad(cost, param) for param in model.stack]

            #updates = [(param, param - args.step * gtheta) for param, gtheta in zip(model.stack, gparams)]

            updates = sgd_updates_adagrad(model.stack, cost)

            print train_batches
            
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
    
