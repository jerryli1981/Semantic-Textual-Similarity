import numpy as np
import random

from scipy.stats import pearsonr

from rnn import depTreeRnnModel
from lstm import depTreeLSTMModel

from mlp import MLP

import theano.tensor as T

from collections import OrderedDict
import theano

import cPickle

from utils import *

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


def train(train_data, dev_data, args, validate=True):

    train_predict(args, trainData=train_data, devData=dev_data, action = 'train', validate=validate) 

def predict(test_data, args):

    train_predict(args, testData=test_data, action = 'predict')

def train_predict(args, action, trainData=None, devData=None, testData=None, epsilon = 1e-16, validate=False):

    import dependency_tree as dep_tree  

    rng = np.random.RandomState(1234)

    relNum = len(dep_tree.loadRelMap())

    word2vecs = dep_tree.loadWord2VecMap()

    if args.repModel == "RNN":
        rep_model = depTreeRnnModel(relNum, args.wvecDim)
    elif args.repModel == "LSTM":
        rep_model = depTreeLSTMModel(args.wvecDim)

    rep_model.initialParams(word2vecs, rng=rng)

    rnn_optimizer = RNN_Optimization(rep_model, alpha=args.step, optimizer=args.optimizer)

    x1_batch = T.fmatrix('x1_batch')  # n * d, the data is presented as one sentence output
    x2_batch = T.fmatrix('x2_batch')  # n * d, the data is presented as one sentence output
    
    y_batch = T.fmatrix('y_batch')  # n * d, the target distribution\

    classifier = MLP(rng=rng,input_1=x1_batch, input_2=x2_batch, n_in=rep_model.wvecDim,
                            n_hidden=args.hiddenDim,n_out=args.outputDim)


    if args.useLearnedModel == "True":
        with open(args.outFile, "rb") as f:
            rep_model.stack = cPickle.load(f)
            classifier.__setstate__(cPickle.load(f))            

    L1_reg=0.00
    L2_reg=0.0001 

    cost = T.mean(classifier.kl_divergence(y_batch)) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    gparams = [ T.grad(cost, param) for param in classifier.params]

    [hw1, hw2, hb, ow, ob] = classifier.params

    delta_x1 = theano.function([x1_batch,x2_batch,y_batch], T.dot(hw1, T.grad(cost,hb)), allow_input_downcast=True)
    delta_x2 = theano.function([x1_batch,x2_batch,y_batch], T.dot(hw2, T.grad(cost,hb)), allow_input_downcast=True)

    act_function = theano.function([x1_batch,x2_batch], classifier.hiddenLayer.output, allow_input_downcast=True)

    learning_rate = args.step
    update_sdg = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
    ]

    update_params_sdg = theano.function(inputs=[x1_batch, x2_batch, y_batch], outputs=cost,
                            updates=update_sdg, allow_input_downcast=True)


    """
    gradt_mlp = [epsilon+T.zeros(W.shape) for W in classifier.params]
    gradt_mlp = [gt+g**2 for gt,g in zip(gradt_mlp, gparams)]

    norm_grad_adagrad =  [g*(1./T.sqrt(gt)) for gt,g in zip(gradt_mlp, gparams)]

    update_adagrad = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, norm_grad_adagrad)
    ]
    """

    grad_updates_adagrad = sgd_updates_adagrad(classifier.params, cost)

    update_params_adagrad = theano.function(inputs=[x1_batch, x2_batch,y_batch], outputs=cost,
                            updates=grad_updates_adagrad, allow_input_downcast=True)



    grad_updates_adadelta = sgd_updates_adadelta(classifier.params, cost)

    """

    gradt_mlp_1 = [epsilon + T.zeros(W.shape) for W in classifier.params]
    gradt_mlp_2 = [epsilon + T.zeros(W.shape) for W in classifier.params]
    eps=1e-6
    rho=0.95

    #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
    gradt_mlp_1 = [rho*gt+(1.0-rho)*(g**2) for gt,g in zip(gradt_mlp_1, gparams)]

    #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
    norm_grad_adadelta = [ -(T.sqrt(gt2+eps) / T.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(gradt_mlp_1, gradt_mlp_2, gparams)]

    #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
    gradt_mlp_2 = [rho*dt + (1.0-rho)*(d ** 2) for dt, d in zip(gradt_mlp_2, norm_grad_adadelta)]

    update_adadelta = [
            (param, param + gparam)
            for param, gparam in zip(classifier.params, norm_grad_adadelta)
    ]
    """
    update_params_adadelta = theano.function(inputs=[x1_batch, x2_batch,y_batch], outputs=cost,
                            updates=grad_updates_adadelta, allow_input_downcast=True)

    #for predict
    mlp_forward = theano.function([x1_batch,x2_batch], classifier.output, allow_input_downcast=True)

    cost_function = theano.function([x1_batch, x2_batch,y_batch], cost, allow_input_downcast=True)
  
    best_dev_score  = 0.
    if action == "train":
        epoch = 0
        print "training model"
        while (epoch < args.epochs):
            epoch = epoch + 1 
            # to compare between different setttings when debugging, but when real training, need remove lambda
            #random.shuffle(trainData, lambda: .5)
            # this is important
            random.shuffle(trainData)

            input_reps_first_train = np.zeros((len(trainData), rep_model.wvecDim))
            input_reps_second_train = np.zeros((len(trainData), rep_model.wvecDim))
            outputs_train = np.zeros((len(trainData), args.outputDim))

            batches = [trainData[idx : idx + args.minibatch] for idx in xrange(0, len(trainData), args.minibatch)]

            for index, batchData in enumerate(batches):

                # this is important to clear the share memory
                rep_model.clearDerivativeSharedMemory()

                targetData = np.zeros((len(batchData), args.outputDim+1))

                for i, (score, item) in enumerate(batchData):
                    sim = score
                    ceil = np.ceil(sim)
                    floor = np.floor(sim)
                    if ceil == floor:
                        targetData[i, floor] = 1
                    else:
                        targetData[i, floor] = ceil-sim 
                        targetData[i, ceil] = sim-floor

                targetData = targetData[:, 1:]

                mul_reps = np.zeros((len(batchData), rep_model.wvecDim))

                sub_reps = np.zeros((len(batchData), rep_model.wvecDim))

                for i, (score, item) in enumerate(batchData): 

                    td = targetData[i]
                    #td += epsilon

                    first_tree_rep = rep_model.forwardProp(item[0])
                    second_tree_rep = rep_model.forwardProp(item[1])
                    mul_rep = first_tree_rep * second_tree_rep
                    sub_rep = np.abs(first_tree_rep-second_tree_rep)

                    input_reps_first_train[i + index*args.minibatch, :] = first_tree_rep
                    input_reps_second_train[i + index*args.minibatch, :] = second_tree_rep
                    outputs_train[i + index*args.minibatch, :] = td

                    mul_reps[i, :] = mul_rep
                    sub_reps[i, :] = sub_rep

                    mul_rep_2d = mul_rep.reshape((1, rep_model.wvecDim))
                    sub_rep_2d = sub_rep.reshape((1, rep_model.wvecDim))
                    td_2d = td.reshape((1, args.outputDim))

                    """"
                    norm = -1.0/args.outputDim
                    sim_grad = norm * targetData[i]
                    sim_grad[sim_grad == -0.] = 0 

                    pd = mlp_forward(mul_rep_2d,sub_rep_2d).reshape(args.outputDim)
                    deltas_softmax = sim_grad * derivative_softmax(pd) #(5,)

                    deltas_hidden = np.dot(classifier.params[3].get_value(), deltas_softmax)

                    act = act_function(mul_rep_2d, sub_rep_2d).reshape(args.hiddenDim)

                    deltas_hidden *= derivative_tanh(act)

                    delta_mul = np.dot(classifier.params[0].get_value(), deltas_hidden.T).reshape(rep_model.wvecDim) #(n_hidden)
                    delta_sub = np.dot(classifier.params[1].get_value(), deltas_hidden.T).reshape(rep_model.wvecDim)
                    """
                    
                    delta_mul = delta_x1(mul_rep_2d, sub_rep_2d, td_2d)  
                    delta_sub = delta_x2(mul_rep_2d, sub_rep_2d, td_2d)

                    delta_rep1_mul = second_tree_rep * delta_mul
                    delta_rep2_mul = first_tree_rep * delta_mul

                    f_s = ((first_tree_rep - second_tree_rep) > 0)
                    s_f = ((second_tree_rep - first_tree_rep) > 0)

                    f_s_a = np.zeros(rep_model.wvecDim)
                    i =0 
                    for x in np.nditer(f_s):
                        if x:
                            f_s_a[i] = 1
                        else:
                            f_s_a[i] = -1
                        i += 1

                    delta_rep1_sub = f_s_a * delta_sub

                    s_f_a = np.zeros(rep_model.wvecDim)

                    i =0 
                    for x in np.nditer(s_f):
                        if x:
                            s_f_a[i] = 1
                        else:
                            s_f_a[i] = -1
                        i += 1

                    delta_rep2_sub = s_f_a * delta_sub
                   
                    rep_model.backProp(item[0], delta_rep1_mul)
                    rep_model.backProp(item[0], delta_rep1_sub)
                    rep_model.backProp(item[1], delta_rep2_mul)
                    rep_model.backProp(item[1], delta_rep2_sub)

                if args.optimizer == 'sgd':

                    update_params_sdg(mul_reps, sub_reps, targetData)

                    update = rep_model.dstack

                    rep_model.stack[1:] = [P-learning_rate*dP for P,dP in zip(rep_model.stack[1:],update[1:])]

                    # handle dictionary update sparsely
                    dL = update[0]
                    for j in range(rep_model.numWords):
                        rep_model.L[:,j] -= learning_rate*dL[j]

                elif args.optimizer == 'adagrad':

                    update_params_adagrad(mul_reps, sub_reps, targetData)
                    rnn_optimizer.adagrad_rnn(rep_model.dstack)

                elif args.optimizer == 'adadelta':
                    update_params_adadelta(mul_reps, sub_reps, targetData)
                    rnn_optimizer.adadelta_rnn(rep_model.dstack)
                else:
                    raise "optimizer is not defined"

                for score, item in batchData:
                    for tree in item:          
                        tree.resetFinished()

            if validate:

                input_reps_first_test = np.zeros((len(devData), rep_model.wvecDim))
                input_reps_second_test = np.zeros((len(devData), rep_model.wvecDim))
                outputs_test = np.zeros((len(devData), args.outputDim))

                targets = np.zeros((len(devData), args.outputDim+1))
      
                for i, (score, item) in enumerate(devData):
                    sim = score
                    ceil = np.ceil(sim)
                    floor = np.floor(sim)
                    if ceil == floor:
                        targets[i, floor] = 1
                    else:
                        targets[i, floor] = ceil-sim
                        targets[i, ceil] = sim-floor

                targets = targets[:, 1:]

                cost = 0
                corrects = []
                guesses = []

                mul_reps = np.zeros((len(devData), rep_model.wvecDim))
                sub_reps = np.zeros((len(devData), rep_model.wvecDim))
                for i, (score, item) in enumerate(devData):

                    td = targets[i]
                    td += epsilon

                    log_td = np.log(td)

                    first_tree_rep= rep_model.forwardProp(item[0])
                    second_tree_rep = rep_model.forwardProp(item[1])
                    mul_rep = first_tree_rep * second_tree_rep
                    sub_rep = np.abs(first_tree_rep-second_tree_rep)

                    input_reps_first_test[i, :] = first_tree_rep
                    input_reps_second_test[i, :] = second_tree_rep
                    outputs_test[i, :] = td

                    mul_reps[i, :] = mul_rep
                    sub_reps[i, :] = sub_rep

                    corrects.append(score)

                pd = mlp_forward(mul_reps,sub_reps)

                predictScores = pd.dot(np.array([1,2,3,4,5]))

                cost = cost_function(mul_reps,sub_reps, targets)
                
                guesses = predictScores.tolist()

                dev_score = pearsonr(corrects,guesses)[0]
            
                for score, item in devData:
                    for tree in item:          
                        tree.resetFinished()
        
                if dev_score > best_dev_score:
                    best_dev_score = dev_score
                    print "iter:%d dev cost: %f dev_score: %f best_dev_score %f"%(epoch, cost, dev_score, best_dev_score)
                    
                    with open("data_keras", "wb") as f:
                        cPickle.dump([input_reps_first_train, input_reps_second_train, outputs_train], f, protocol=cPickle.HIGHEST_PROTOCOL)
                        cPickle.dump([input_reps_first_test, input_reps_second_test, outputs_test], f, protocol=cPickle.HIGHEST_PROTOCOL)
    
                else:
                    print "iter:%d dev cost: %f dev_score: %f"%(epoch, cost, dev_score)


            with open(args.outFile, "wb") as f:
                cPickle.dump(rep_model.stack, f, protocol=cPickle.HIGHEST_PROTOCOL)
                cPickle.dump(classifier.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)  

    if action == "predict":

        print "predict model"

        if args.repModel == "RNN":
            rep_model = depTreeRnnModel(relNum, args.wvecDim)
        elif args.repModel == "LSTM":
            rep_model = depTreeLSTMModel(args.wvecDim)

        rep_model.initialParams(word2vecs, rng=rng)
        classifier = MLP(rng=rng,input_1=x1_batch, input_2=x2_batch, n_in=args.wvecDim,
                        n_hidden=args.hiddenDim,n_out=args.outputDim)

        with open(args.outFile, "rb") as f:
            rep_model.stack = cPickle.load(f)
            classifier.__setstate__(cPickle.load(f)) 

        testData = testData[:10]

        targets = np.zeros((len(testData), args.outputDim+1))

        for i, (score, item) in enumerate(testData):
            sim = score
            ceil = np.ceil(sim)
            floor = np.floor(sim)
            if ceil == floor:
                targets[i, floor] = 1
            else:
                targets[i, floor] = ceil-sim
                targets[i, ceil] = sim-floor

        targets = targets[:, 1:]

        corrects = []
        guesses = []

        mul_reps = np.zeros((len(testData), rep_model.wvecDim))
        sub_reps = np.zeros((len(testData), rep_model.wvecDim))

        for i, (score, item) in enumerate(testData):

            td = targets[i]
            td += epsilon

            log_td = np.log(td)

            first_tree_rep= rep_model.forwardProp(item[0])
            second_tree_rep = rep_model.forwardProp(item[1])
            mul_rep = first_tree_rep * second_tree_rep
            sub_rep = np.abs(first_tree_rep-second_tree_rep)

            mul_rep_2d = mul_rep.reshape((1, rep_model.wvecDim))
            sub_rep_2d = sub_rep.reshape((1, rep_model.wvecDim))

            mul_reps[i, :] = mul_rep
            sub_reps[i, :] = sub_rep
    
            corrects.append(score)

        pd = mlp_forward(mul_reps,sub_reps)

        predictScores = pd.dot(np.array([1,2,3,4,5]))
        
        guesses = predictScores.tolist()

        test_score = pearsonr(corrects,guesses)[0]

        print "test score: %f"%(test_score)

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


