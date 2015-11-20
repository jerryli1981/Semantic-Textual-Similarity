import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from scipy.stats import pearsonr

from rnn import depTreeRnnModel
from lstm import depTreeLSTMModel

from mlp import MLP

from collections import defaultdict, OrderedDict

from utils import iterate_minibatches_tree, loadWord2VecMap,read_dataset_tree

import autograd.numpy as auto_grad_np
from autograd import grad, elementwise_grad

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
        exp_sqr_grads[param] = theano.shared(value=empty,name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=empty, name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        updates[param] = stepped_param      
    return updates 

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
    return updates 

def mul(first_tree_rep, second_tree_rep):

    return auto_grad_np.multiply(first_tree_rep, second_tree_rep)

def abs_sub(first_tree_rep, second_tree_rep, epsilon = 1e-16):

    return auto_grad_np.abs(first_tree_rep-second_tree_rep + epsilon)

def train(train_data, dev_data, args, validate=True):

    train_predict(args, trainData=train_data, devData=dev_data, action = 'train', validate=validate) 


def build_network(args, wordEmbeddings, L1_reg=0.00, L2_reg=0.0001):

    print("Building model ...")

    rng = np.random.RandomState(1234)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    rel_vocab_path = os.path.join(sick_dir, 'rel_vocab.txt')

    rels = defaultdict(int)

    with open(rel_vocab_path, 'r') as f:
        for tok in f:
            rels[tok.rstrip('\n')] += 1


    if args.repModel == "RNN":
        rep_model = depTreeRnnModel(len(rels), args.wvecDim)
    elif args.repModel == "LSTM":
        rep_model = depTreeLSTMModel(args.wvecDim)

    rep_model.initialParams(wordEmbeddings, rng=rng)

    rnn_optimizer = RNN_Optimization(rep_model, alpha=args.step, optimizer=args.optimizer)


    x = T.fmatrix('x')  # n * d, the data is presented as one sentence output    
    y = T.fmatrix('y')  # n * d, the target distribution\

    classifier = MLP(rng=rng,input=x, n_in=2*rep_model.wvecDim, n_hidden=args.hiddenDim,n_out=args.outputDim)

    cost = T.mean(classifier.kl_divergence(y)) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    gparams = [ T.grad(cost, param) for param in classifier.params]

    hw = classifier.params[0]
    hb = classifier.params[1]
    delta_x = theano.function([x,y], T.dot(hw, T.grad(cost, hb)), allow_input_downcast=True)
    
    if args.optimizer == "sgd":

        update_sdg = [
            (param, param - args.step * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]

        update_params_theano = theano.function(inputs=[x,y], outputs=cost,
                                updates=update_sdg, allow_input_downcast=True)
    elif args.optimizer == "adagrad":

        grad_updates_adagrad = sgd_updates_adagrad(classifier.params, cost)

        update_params_theano = theano.function(inputs=[x,y], outputs=cost,
                                updates=grad_updates_adagrad, allow_input_downcast=True)

    elif args.optimizer == "adadelta":

        grad_updates_adadelta = sgd_updates_adadelta(classifier.params, cost)

        update_params_theano = theano.function(inputs=[x,y], outputs=cost,
                                updates=grad_updates_adadelta, allow_input_downcast=True)


    cost_and_prob = theano.function([x, y], [cost, classifier.output], allow_input_downcast=True)

    return rep_model, rnn_optimizer, update_params_theano, delta_x, cost_and_prob


def train(args, rep_model, rnn_optimizer, update_params_theano, delta_x, batchData):

    # this is important to clear the share memory
    rep_model.clearDerivativeSharedMemory()

    l_trees, r_trees, Y_labels, Y_scores, Y_scores_pred = batchData

    vec_feats = np.zeros((len(l_trees), 2*args.wvecDim))

    for i, (l_t, r_t, td) in enumerate(zip(l_trees, r_trees, Y_scores_pred)):


        first_tree_rep = rep_model.forwardProp(l_t)
        second_tree_rep = rep_model.forwardProp(r_t)

        mul_rep = mul(first_tree_rep, second_tree_rep)
        sub_rep = abs_sub(first_tree_rep, second_tree_rep)

        vec_feat = np.concatenate((mul_rep, sub_rep))
        vec_feats[i] = vec_feat

        vec_feat_2d = vec_feat.reshape((1, 2*rep_model.wvecDim))

        td_2d = td.reshape((1, args.outputDim))

        delta = delta_x(vec_feat_2d, td_2d)
        delta_mul = delta[:rep_model.wvecDim]
        delta_sub = delta[rep_model.wvecDim:]

        mul_grad_1 = elementwise_grad(mul, argnum=0) 
        mul_grad_2 = elementwise_grad(mul, argnum=1)

        first_mul_grad = mul_grad_1(first_tree_rep,second_tree_rep)
        second_mul_grad = mul_grad_2(first_tree_rep,second_tree_rep)

        delta_rep1_mul =  first_mul_grad * delta_mul
        delta_rep2_mul =  second_mul_grad * delta_mul

        sub_grad_1 = elementwise_grad(abs_sub, argnum=0) 
        sub_grad_2 = elementwise_grad(abs_sub, argnum=1) 

        first_sub_grad = sub_grad_1(first_tree_rep,second_tree_rep)
        second_sub_grad = sub_grad_2(first_tree_rep,second_tree_rep)

        delta_rep1_sub = first_sub_grad * delta_sub
        delta_rep2_sub =  second_sub_grad * delta_sub

        rep_model.backProp(l_t, delta_rep1_mul)
        rep_model.backProp(l_t, delta_rep1_sub)
        rep_model.backProp(r_t, delta_rep2_mul)
        rep_model.backProp(r_t, delta_rep2_sub)

    if args.optimizer == 'sgd':

        cost = update_params_theano(mul_reps, sub_reps, Y_scores_pred)

        update = rep_model.dstack

        rep_model.stack[1:] = [P-learning_rate*dP for P,dP in zip(rep_model.stack[1:],update[1:])]

        # handle dictionary update sparsely
        """
        dL = update[0]
        for j in range(rep_model.numWords):
            rep_model.L[:,j] -= learning_rate*dL[j]
        """

    elif args.optimizer == 'adagrad':

        cost = update_params_theano(vec_feats, Y_scores_pred)
        
        rnn_optimizer.adagrad_rnn(rep_model.dstack)

    elif args.optimizer == 'adadelta':

        cost = update_params_theano(mul_reps, sub_reps, Y_scores_pred)
        rnn_optimizer.adadelta_rnn(rep_model.dstack)

    for l_t, r_t  in zip(l_trees, r_trees):      
        l_t.resetFinished()
        r_t.resetFinished()

    return cost

def validate(args, rep_model, cost_and_prob, devData):

    l_trees_d, r_trees_d, Y_labels_d, Y_scores_d, Y_scores_pred_d = devData

    cost = 0
    corrects = []
    guesses = []

    #mul_reps = np.zeros((len(devData), rep_model.wvecDim))
    #sub_reps = np.zeros((len(devData), rep_model.wvecDim))

    vec_feats = np.zeros((len(l_trees_d), 2*rep_model.wvecDim))

    for i, (l_t, r_t, score, td) in enumerate(zip(l_trees_d, r_trees_d, Y_scores_d, Y_scores_pred_d)):
    #for i, (score, item) in enumerate(devData):

        #td = targets[i]
        #td += epsilon

        log_td = np.log(td)

        first_tree_rep= rep_model.forwardProp(l_t)
        second_tree_rep = rep_model.forwardProp(r_t)
        mul_rep = first_tree_rep * second_tree_rep
        sub_rep = np.abs(first_tree_rep-second_tree_rep)

        #input_reps_first_test[i, :] = first_tree_rep
        #input_reps_second_test[i, :] = second_tree_rep
        #outputs_test[i, :] = td

        #mul_reps[i, :] = mul_rep
        #sub_reps[i, :] = sub_rep

        vec_feat = np.concatenate((mul_rep, sub_rep))
        vec_feats[i] = vec_feat

        corrects.append(score)

    cost, pd = cost_and_prob(vec_feats,Y_scores_pred_d)

    for l_t, r_t  in zip(l_trees_d, r_trees_d):      
        l_t.resetFinished()
        r_t.resetFinished()

    return cost, pd

class RNN_Optimization:

    def __init__(self, rep_model, alpha=0.01, epsilon = 1e-16, optimizer='sgd'):

        self.learning_rate = alpha # learning rate
        self.optimizer = optimizer
        self.rep_model = rep_model
        
        if self.optimizer == 'adagrad':

            self.gradt_rnn = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]

        elif self.optimizer =="adadelta":

            self.gradt_rnn_1 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]
            self.gradt_rnn_2 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]
        
    def adagrad_rnn(self, grad):

        # trace = trace+grad.^2
        self.gradt_rnn[1:] = [gt+g**2 for gt,g in zip(self.gradt_rnn[1:],grad[1:])]
        # update = grad.*trace.^(-1/2)
        dparam =  [g*(1./np.sqrt(gt)) for gt,g in zip(self.gradt_rnn[1:],grad[1:])]

        self.rep_model.stack[1:] = [P-self.learning_rate*dP for P,dP in zip(self.rep_model.stack[1:],dparam)]


        """
        # handle dictionary separately
        dL = grad[0]
        dLt = self.gradt_rnn[0]
        for j in range(self.rep_model.numWords):
            #dLt[:,j] = dLt[:,j] + dL[:,j]**2
            #dL[:,j] = dL[:,j] * (1./np.sqrt(dLt[:,j]))
            dLt[:,j] = dLt[:,j] + dL[j]**2
            dL[j] = dL[j] * (1./np.sqrt(dLt[:,j]))

        # handle dictionary update sparsely
        for j in range(self.rep_model.numWords):
            #self.rep_model.L[:,j] -= self.learning_rate*dL[:,j]
            self.rep_model.L[:,j] -= self.learning_rate*dL[j]
        """

    def adadelta_rnn(self, grad, eps=1e-6, rho=0.95):

        #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
        self.gradt_rnn_1[1:] = [rho*gt+(1.0-rho) * (g ** 2) for gt,g in zip(self.gradt_rnn_1[1:], grad[1:])]

        #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
        dparam = [ - (np.sqrt(gt2+eps) / np.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(self.gradt_rnn_1[1:], self.gradt_rnn_2[1:], grad[1:])]
        
        self.rep_model.stack[1:] = [P+ dP for P,dP in zip(self.rep_model.stack[1:],dparam)]

        dL = grad[0]
        dLt_1 = self.gradt_rnn_1[0]
        dLt_2 = self.gradt_rnn_2[0]

        for j in range(self.rep_model.numWords):
            #dLt_1[:,j] = rho*dLt_1[:,j]+(1.0-rho)*(dL[:,j] ** 2)
            #dL[:,j] = -( np.sqrt(dLt_2[:,j]+eps)/ np.sqrt(dLt_1[:,j]+eps) ) * dL[:,j]
            dLt_1[:,j] = rho*dLt_1[:,j]+(1.0-rho)*(dL[j] ** 2)
            dL[j] = -( np.sqrt(dLt_2[:,j]+eps)/ np.sqrt(dLt_1[:,j]+eps) ) * dL[j]

            #update
            #dLt_2[:,j] = rho*dLt_2[:,j] + (1.0-rho)*(dL[:,j] ** 2)
            dLt_2[:,j] = rho*dLt_2[:,j] + (1.0-rho)*(dL[j] ** 2)

        for j in range(self.rep_model.numWords):
            #self.rep_model.L[:,j] += dL[:,j]
            self.rep_model.L[:,j] += dL[j]

        #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
        self.gradt_rnn_2[1:] = [rho*dt + (1.0-rho)*( d** 2) for dt, d in zip(self.gradt_rnn_2[1:], dparam)]

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=25)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=10)
    parser.add_argument("--step",dest="step",type=float,default=0.05)
    parser.add_argument("--outputDim",dest="outputDim",type=int,default=5)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--wvecDim",dest="wvecDim",type=int,default=150)
    parser.add_argument("--repModel",dest="repModel",type=str,default="RNN")
    args = parser.parse_args()

     # Load the dataset
    print("Loading data...")
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    
    trainTrees = read_dataset_tree(sick_dir, "train")
    devTrees = read_dataset_tree(sick_dir, "dev")
    testTrees = read_dataset_tree(sick_dir, "test")

    rep_model, rnn_optimizer, update_params_theano, delta_x, cost_and_prob = build_network(args, wordEmbeddings)

    print("Starting training...")
    best_val_acc = 0
    best_val_pearson = 0
    for epoch in range(args.epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches_tree(trainTrees, args.minibatch, shuffle=True):

            train_err += train(args, rep_model, rnn_optimizer, update_params_theano, delta_x, batch)
            train_batches += 1
 
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_pearson = 0

        for batch in iterate_minibatches_tree(devTrees, args.minibatch):

            _, _, _, scores, _= batch

            err, preds = validate(args, rep_model, cost_and_prob, batch)

            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            val_pearson += pearson_score 

            val_err += err
            
            val_batches += 1

            
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


        val_score = val_pearson / val_batches * 100
        print("  validation pearson:\t\t{:.2f} %".format(
            val_pearson / val_batches * 100))
        if best_val_pearson < val_score:
            best_val_pearson = val_score


    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_pearson = 0
    test_batches = 0
    for batch in iterate_minibatches_tree(testTrees, args.minibatch):

        _, _, _, scores, _= batch

        err, preds = validate(args, rep_model, cost_and_prob, batch)

        predictScores = preds.dot(np.array([1,2,3,4,5]))
        guesses = predictScores.tolist()
        scores = scores.tolist()
        pearson_score = pearsonr(scores,guesses)[0]
        test_pearson += pearson_score 

        test_err += err
        
        test_batches += 1


    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    print("  Best validate perason:\t\t{:.2f} %".format(best_val_pearson))
    print("  test pearson:\t\t{:.2f} %".format(
        test_pearson / test_batches * 100))

