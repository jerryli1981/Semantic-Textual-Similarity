import numpy as np
import random

from scipy.stats import pearsonr

from rnn import depTreeRnnModel
from lstm import depTreeLSTMModel

from mlp import MLP

import theano.tensor as T
import theano

import cPickle

from utils import *

def initial_RepModel(rng, dep_tree, model, wvecDim, outputFile, startFromEariler):

    relNum = len(dep_tree.loadRelMap())

    word2vecs = dep_tree.loadWord2VecMap()

    if model == "RNN":
        rep_model = depTreeRnnModel(relNum, wvecDim)
    elif model == "LSTM":
        rep_model = depTreeLSTMModel(wvecDim)

    rep_model.initialParams(word2vecs, rng=rng)

    if startFromEariler == "True":
        with open(outputFile, "rb") as f:
            rep_model.stack = cPickle.load(f)
            _ = cPickle.load(f)

    return rep_model


def adagrad_mlp(gradt_mlp, grad):

    gradt_mlp = [gt+g**2 for gt,g in zip(gradt_mlp, grad)]

    updates =  [g*(1./T.sqrt(gt)) for gt,g in zip(gradt_mlp, grad)]

    return updates

def adadelta_mlp(gradt_mlp_1, gradt_mlp_2, grad, eps=1e-6, rho=0.95):

    #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
    gradt_mlp_1 = [rho*gt+(1.0-rho)*(g**2) for gt,g in zip(gradt_mlp_1, grad)]

    #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
    dparams = [ -(T.sqrt(gt2+eps) / T.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(gradt_mlp_1, gradt_mlp_2, grad)]

    #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
    gradt_mlp_2 = [rho*dt + (1.0-rho)*(d ** 2) for dt, d in zip(gradt_mlp_2, dparams)]

    return dparams

def train_with_theano_mlp(rng, rnn_optimizer, rep_model, trainData, devData, batchSize, epochs, 
    hiddenDim, outputDim, learning_rate, optimizer, startFromEariler, outputFile, action, epsilon = 1e-16):


    x1_batch = T.fmatrix('x1_batch')  # n * d, the data is presented as one sentence output
    x2_batch = T.fmatrix('x2_batch')  # n * d, the data is presented as one sentence output
    y_batch = T.fmatrix('y_batch')  # n * d, the target distribution\


    classifier = MLP(rng=rng,input_1=x1_batch, input_2=x2_batch, n_in=rep_model.wvecDim,
                            n_hidden=hiddenDim,n_out=outputDim)

    if startFromEariler == "True":
        with open(outputFile, "rb") as f:
            _ = cPickle.load(f)
            classifier.__setstate__(cPickle.load(f))

    L1_reg=0.00
    L2_reg=0.0001 

    cost = classifier.kl_divergence(y_batch) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    gparams = [ T.grad(cost, param) for param in classifier.params]

    [hw1, hw2, hb, ow, ob] = classifier.params

    delta_hw1 = theano.function([x1_batch,x2_batch,y_batch], T.dot(hw1, T.grad(cost,hb)), allow_input_downcast=True)
    delta_hw2 = theano.function([x1_batch,x2_batch,y_batch], T.dot(hw2, T.grad(cost,hb)), allow_input_downcast=True)
    delta_ob = theano.function([x1_batch,x2_batch,y_batch], T.grad(cost,ob), allow_input_downcast=True)

    act_function = theano.function([x1_batch,x2_batch], classifier.hiddenLayer.output, allow_input_downcast=True)

    update_sdg = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
    ]

    update_params_sdg = theano.function(inputs=[x1_batch, x2_batch, y_batch], outputs=cost,
                            updates=update_sdg, allow_input_downcast=True)


    
    gradt_mlp = [epsilon + T.zeros(W.shape) for W in classifier.params]
    gradt_mlp_1 = [epsilon + T.zeros(W.shape) for W in classifier.params]
    gradt_mlp_2 = [epsilon + T.zeros(W.shape) for W in classifier.params]


    update_adagrad = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, adagrad_mlp(gradt_mlp, gparams))
    ]

    update_params_adagrad = theano.function(inputs=[x1_batch, x2_batch,y_batch], outputs=cost,
                            updates=update_adagrad, allow_input_downcast=True)


    update_adadelta = [
            (param, param + gparam)
            for param, gparam in zip(classifier.params, adadelta_mlp(gradt_mlp_1, gradt_mlp_2, gparams))
    ]

    update_params_adadelta = theano.function(inputs=[x1_batch, x2_batch,y_batch], outputs=cost,
                            updates=update_adadelta, allow_input_downcast=True)

    #for predict
    mlp_forward = theano.function([x1_batch,x2_batch], classifier.output, allow_input_downcast=True)

    best_dev_score  = 0.
    epoch =0
    while (epoch < epochs):
        epoch = epoch + 1 
        # to compare between different setttings when debugging, but when real training, need remove lambda
        #random.shuffle(trainData, lambda: .5)
        # this is important
        random.shuffle(trainData)

        batches = [trainData[idx : idx + batchSize] for idx in xrange(0, len(trainData), batchSize)]

        for index, batchData in enumerate(batches):

            """""""""""""""""""""""""""""""""
            belwo are costAndGrad_theano_grad
            """""""""""""""""""""""""""""""""
            # this is important to clear the share memory
            rep_model.clearDerivativeSharedMemory()

            targetData = np.zeros((len(batchData), outputDim+1))

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

                mul_reps[i, :] = mul_rep
                sub_reps[i, :] = sub_rep

                mul_rep_2d = mul_rep.reshape((1, rep_model.wvecDim))
                sub_rep_2d = sub_rep.reshape((1, rep_model.wvecDim))
                td_2d = td.reshape((1, outputDim))

                
                norm = -1.0/outputDim
                sim_grad = norm * targetData[i]
                sim_grad[sim_grad == -0.] = 0 

                pd = mlp_forward(mul_rep_2d,sub_rep_2d).reshape(outputDim)
                deltas_softmax = sim_grad * derivative_softmax(pd) #(5,)

                deltas_hidden = np.dot(classifier.params[3].get_value(), deltas_softmax)

                act = act_function(mul_rep_2d, sub_rep_2d).reshape(hiddenDim)

                deltas_hidden *= derivative_tanh(act)

                delta_mul = np.dot(classifier.params[0].get_value(), deltas_hidden.T).reshape(rep_model.wvecDim) #(n_hidden)
                delta_sub = np.dot(classifier.params[1].get_value(), deltas_hidden.T).reshape(rep_model.wvecDim)

                
                #delta_mul = self.delta_hw1(mul_rep_2d, sub_rep_2d, td_2d)  
                #delta_sub = self.delta_hw2(mul_rep_2d, sub_rep_2d, td_2d)

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

            if optimizer == 'sgd':

                update_params_sdg(mul_reps, sub_reps, targetData)

                update = rep_model.dstack

                rep_model.stack[1:] = [P-learning_rate*dP for P,dP in zip(rep_model.stack[1:],update[1:])]

                # handle dictionary update sparsely
                dL = update[0]
                for j in range(rep_model.numWords):
                    rep_model.L[:,j] -= learning_rate*dL[:,j]

            elif optimizer == 'adagrad':

                update_params_adagrad(mul_reps, sub_reps, targetData)
                rnn_optimizer.adagrad_rnn(rep_model.dstack)

            elif optimizer == 'adadelta':
                update_params_adadelta(mul_reps, sub_reps, targetData)
                rnn_optimizer.adadelta_rnn(rep_model.dstack)
            else:
                raise "optimizer is not defined"

            for score, item in batchData:
                for tree in item:          
                    tree.resetFinished()

        if action=="test":

            targets = np.zeros((len(devData), outputDim+1))
  
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

            for i, (score, item) in enumerate(devData):

                td = targets[i]
                td += epsilon

                log_td = np.log(td)

                first_tree_rep= rep_model.forwardProp(item[0])
                second_tree_rep = rep_model.forwardProp(item[1])
                mul_rep = first_tree_rep * second_tree_rep
                sub_rep = np.abs(first_tree_rep-second_tree_rep)

                mul_rep_2d = mul_rep.reshape((1, rep_model.wvecDim))
                sub_rep_2d = sub_rep.reshape((1, rep_model.wvecDim))
        
                pd = mlp_forward(mul_rep_2d,sub_rep_2d)

                predictScore = pd.reshape(outputDim).dot(np.array([1,2,3,4,5]))

                predictScore = float("{0:.2f}".format(predictScore))

                loss = np.dot(td, log_td-np.log(pd.reshape(outputDim))) / outputDim

                cost += loss
                corrects.append(score)
                guesses.append(predictScore)

            for score, item in devData:
                for tree in item:          
                    tree.resetFinished()
    
            cost = cost/len(devData)
            dev_score = pearsonr(corrects,guesses)[0]

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                print "iter:%d cost: %f dev_score: %f best_dev_score %f"%(epoch, cost, dev_score, best_dev_score)
                with open(outputFile, "wb") as f:
                    cPickle.dump(rep_model.stack, f, protocol=cPickle.HIGHEST_PROTOCOL)
                    cPickle.dump(classifier.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
            else:
                print "iter:%d cost: %f dev_score: %f"%(epoch, cost, dev_score)


class RNN_Optimization:

    def __init__(self, rep_model, alpha=0.01, epsilon = 1e-16, optimizer='sgd'):

        self.learning_rate = alpha # learning rate
        self.optimizer = optimizer
        self.rep_model = rep_model
        
        print "Using",self.optimizer

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


