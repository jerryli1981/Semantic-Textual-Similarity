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

class Optimization:

    def __init__(self, outputFile, alpha=0.01, optimizer='sgd'):

        self.learning_rate = alpha # learning rate
        self.optimizer = optimizer
        self.rng = np.random.RandomState(1234)
        self.outputFile = outputFile
        print "Using",self.optimizer

    def initial_RepModel(self, dep_tree, model, wvecDim, epsilon = 1e-16, startFromEariler=False):

        relNum = len(dep_tree.loadRelMap())

        word2vecs = dep_tree.loadWord2VecMap()

        if model == "RNN":
            self.rep_model = depTreeRnnModel(relNum, wvecDim)
        elif model == "LSTM":
            self.rep_model = depTreeLSTMModel(wvecDim)

        self.rep_model.initialParams(word2vecs, rng=self.rng)

        if startFromEariler:
            with open(self.outputFile, "rb") as f:
                self.rep_model.stack = cPickle.load(f)
                _ = cPickle.load(f)

        if self.optimizer == 'adagrad':

            self.gradt_rnn = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]

        elif self.optimizer =="adadelta":

            self.gradt_rnn_1 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]
            self.gradt_rnn_2 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]

        
    def initial_theano_mlp(self, hiddenDim, outputDim, epsilon = 1e-16, startFromEariler=False):

        self.outputDim = outputDim
        self.hiddenDim = hiddenDim

        x1_batch = T.fmatrix('x1_batch')  # n * d, the data is presented as one sentence output
        x2_batch = T.fmatrix('x2_batch')  # n * d, the data is presented as one sentence output
        y_batch = T.fmatrix('y_batch')  # n * d, the target distribution\

        self.classifier = MLP(rng=self.rng,input_1=x1_batch, input_2=x2_batch, n_in=self.rep_model.wvecDim,
                                n_hidden=hiddenDim,n_out=outputDim)

        if startFromEariler:
            with open(self.outputFile, "rb") as f:
                _ = cPickle.load(f)
                hw1, hw2, hb = cPickle.load(f)
                ow, ob = cPickle.load(f)

            self.classifier = MLP(rng=self.rng,input_1=x1_batch, input_2=x2_batch, n_in=self.rep_model.wvecDim,
                                n_hidden=hiddenDim,n_out=outputDim,activation=activation, 
                                hw1=hw1, hw2=hw2, hb=hb, ow=ow, ob=ob)


        L1_reg=0.00
        L2_reg=0.0001 

        #cost = self.classifier.kl_divergence(y_batch) + L1_reg * self.classifier.L1 + L2_reg * self.classifier.L2_sqr

        cost = T.mean(classifier.kl_divergence(y_batch)) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

        [hw1, hw2, hb, ow, ob] = self.classifier.hiddenLayer.params + self.classifier.logRegressionLayer.params

        self.classifier.params = self.classifier.hiddenLayer.params + self.classifier.logRegressionLayer.params

        gparams = [ T.grad(cost, param) for param in self.classifier.params]

        self.delta_hw1 = theano.function([x1_batch,x2_batch,y_batch], T.dot(hw1, T.grad(cost,hb)), allow_input_downcast=True)
        self.delta_hw2 = theano.function([x1_batch,x2_batch,y_batch], T.dot(hw2, T.grad(cost,hb)), allow_input_downcast=True)
        self.delta_ob = theano.function([x1_batch,x2_batch,y_batch], T.grad(cost,ob), allow_input_downcast=True)

        act = self.classifier.hiddenLayer.output
        self.act_function = theano.function([x1_batch,x2_batch], act, allow_input_downcast=True)

        update_sdg = [
                (param, param - self.learning_rate * gparam)
                for param, gparam in zip(self.classifier.params, gparams)
        ]

        self.update_params_sdg = theano.function(inputs=[x1_batch, x2_batch, y_batch], outputs=cost,
                                updates=update_sdg, allow_input_downcast=True)


        
        self.gradt_mlp = [epsilon + T.zeros(W.shape) for W in self.classifier.params]
        self.gradt_mlp_1 = [epsilon + T.zeros(W.shape) for W in self.classifier.params]
        self.gradt_mlp_2 = [epsilon + T.zeros(W.shape) for W in self.classifier.params]


        grad_updates_adagrad = self.adagrad_mlp(gparams)

        update_adagrad = [
                (param, param - self.learning_rate * gparam)
                for param, gparam in zip(self.classifier.params, grad_updates_adagrad)
        ]

        self.update_params_adagrad = theano.function(inputs=[x1_batch, x2_batch,y_batch], outputs=cost,
                                updates=update_adagrad, allow_input_downcast=True)

        grad_updates_adadelta = self.adadelta_mlp(gparams)

        update_adadelta = [
                (param, param + gparam)
                for param, gparam in zip(self.classifier.params, grad_updates_adadelta)
        ]

        self.update_params_adadelta = theano.function(inputs=[x1_batch, x2_batch,y_batch], outputs=cost,
                                updates=update_adadelta, allow_input_downcast=True)


        #for predict
        self.mlp_forward = theano.function([x1_batch,x2_batch], self.classifier.output, allow_input_downcast=True)



        
    def train_with_theano_mlp(self, trainData, batchSize):

        # to compare between different setttings when debugging, but when real training, need remove lambda
        #random.shuffle(trainData, lambda: .5)
        # this is important
        random.shuffle(trainData)

        batches = [trainData[idx : idx + batchSize] for idx in xrange(0, len(trainData), batchSize)]

        for index, batchData in enumerate(batches):

            rep_model_dstack, mul_reps, sub_reps, targetData = self.costAndGrad_theano_grad(batchData)

            if self.optimizer == 'sgd':

                self.update_params_sdg(mul_reps, sub_reps, targetData)

                update = rep_model_dstack

                self.rep_model.stack[1:] = [P-self.learning_rate*dP for P,dP in zip(self.rep_model.stack[1:],update[1:])]

                # handle dictionary update sparsely
                dL = update[0]
                for j in range(self.rep_model.numWords):
                    self.rep_model.L[:,j] -= self.learning_rate*dL[:,j]

            elif self.optimizer == 'adagrad':

                self.update_params_adagrad(mul_reps, sub_reps, targetData)
                self.adagrad_rnn(rep_model_dstack)

            elif self.optimizer == 'adadelta':
                self.update_params_adadelta(mul_reps, sub_reps, targetData)
                self.adadelta_rnn(rep_model_dstack)
            else:
                raise "optimizer is not defined"

            for score, item in batchData:
                for tree in item:          
                    tree.resetFinished()


    def costAndGrad_theano_grad(self, trainData, epsilon = 1e-16):

        # this is important to clear the share memory
        self.rep_model.clearDerivativeSharedMemory()

        targetData = np.zeros((len(trainData), self.outputDim+1))

        for i, (score, item) in enumerate(trainData):
            sim = score
            ceil = np.ceil(sim)
            floor = np.floor(sim)
            if ceil == floor:
                targetData[i, floor] = 1
            else:
                targetData[i, floor] = ceil-sim 
                targetData[i, ceil] = sim-floor

        targetData = targetData[:, 1:]

        mul_reps = np.zeros((len(trainData), self.rep_model.wvecDim))
        sub_reps = np.zeros((len(trainData), self.rep_model.wvecDim))

        for i, (score, item) in enumerate(trainData): 

            td = targetData[i]
            #td += epsilon

            first_tree_rep = self.rep_model.forwardProp(item[0])
            second_tree_rep = self.rep_model.forwardProp(item[1])
            mul_rep = first_tree_rep * second_tree_rep
            sub_rep = np.abs(first_tree_rep-second_tree_rep)

            mul_reps[i, :] = mul_rep
            sub_reps[i, :] = sub_rep

            mul_rep_2d = mul_rep.reshape((1, self.rep_model.wvecDim))
            sub_rep_2d = sub_rep.reshape((1, self.rep_model.wvecDim))
            td_2d = td.reshape((1, self.outputDim))

            
            norm = -1.0/self.outputDim
            sim_grad = norm * targetData[i]
            sim_grad[sim_grad == -0.] = 0 

            pd = self.mlp_forward(mul_rep_2d,sub_rep_2d).reshape(self.outputDim)
            deltas_softmax = sim_grad * derivative_softmax(pd) #(5,)

            deltas_hidden = np.dot(self.classifier.params[3].eval(), deltas_softmax)

            act = self.act_function(mul_rep_2d, sub_rep_2d).reshape(self.hiddenDim)

            if self.activation == "tanh":
                deltas_hidden *= derivative_tanh(act)
            elif self.activation == "sigmoid":
                deltas_hidden *= derivative_sigmoid(act)
            else:
                raise "incorrect activation function"
            delta_mul = np.dot(self.classifier.params[0].eval(), deltas_hidden.T).reshape(self.rep_model.wvecDim) #(n_hidden)
            delta_sub = np.dot(self.classifier.params[1].eval(), deltas_hidden.T).reshape(self.rep_model.wvecDim)

            
            #delta_mul = self.delta_hw1(mul_rep_2d, sub_rep_2d, td_2d)  
            #delta_sub = self.delta_hw2(mul_rep_2d, sub_rep_2d, td_2d)

            delta_rep1_mul = second_tree_rep * delta_mul
            delta_rep2_mul = first_tree_rep * delta_mul

            f_s = ((first_tree_rep - second_tree_rep) > 0)
            s_f = ((second_tree_rep - first_tree_rep) > 0)

            f_s_a = np.zeros(self.rep_model.wvecDim)
            i =0 
            for x in np.nditer(f_s):
                if x:
                    f_s_a[i] = 1
                else:
                    f_s_a[i] = -1
                i += 1

            delta_rep1_sub = f_s_a * delta_sub

            s_f_a = np.zeros(self.rep_model.wvecDim)

            i =0 
            for x in np.nditer(s_f):
                if x:
                    s_f_a[i] = 1
                else:
                    s_f_a[i] = -1
                i += 1

            delta_rep2_sub = s_f_a * delta_sub
           
            self.rep_model.backProp(item[0], delta_rep1_mul)
            self.rep_model.backProp(item[0], delta_rep1_sub)
            self.rep_model.backProp(item[1], delta_rep2_mul)
            self.rep_model.backProp(item[1], delta_rep2_sub)


        return self.rep_model.dstack, mul_reps, sub_reps, targetData
        
    def adagrad_mlp(self, grad):

        self.gradt_mlp = [gt+g**2 for gt,g in zip(self.gradt_mlp, grad)]

        updates =  [g*(1./T.sqrt(gt)) for gt,g in zip(self.gradt_mlp, grad)]

        return updates

    def adadelta_mlp(self, grad, eps=1e-6, rho=0.95):

        #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
        self.gradt_mlp_1 = [rho*gt+(1.0-rho)*(g**2) for gt,g in zip(self.gradt_mlp_1, grad)]
 
        #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
        dparams = [ -(T.sqrt(gt2+eps) / T.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(self.gradt_mlp_1, self.gradt_mlp_2, grad)]

        #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
        self.gradt_mlp_2 = [rho*dt + (1.0-rho)*(d ** 2) for dt, d in zip(self.gradt_mlp_2, dparams)]

        return dparams

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

    def predict(self, trees, epsilon=1e-16):

        targets = np.zeros((len(trees), self.outputDim+1))
  
        for i, (score, item) in enumerate(trees):
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

        for i, (score, item) in enumerate(trees):

            td = targets[i]
            td += epsilon

            log_td = np.log(td)

            first_tree_rep= self.rep_model.forwardProp(item[0])
            second_tree_rep = self.rep_model.forwardProp(item[1])
            mul_rep = first_tree_rep * second_tree_rep
            sub_rep = np.abs(first_tree_rep-second_tree_rep)

            mul_rep_2d = mul_rep.reshape((1, self.rep_model.wvecDim))
            sub_rep_2d = sub_rep.reshape((1, self.rep_model.wvecDim))
    
            pd = self.mlp_forward(mul_rep_2d,sub_rep_2d)

            predictScore = pd.reshape(self.outputDim).dot(np.array([1,2,3,4,5]))

            predictScore = float("{0:.2f}".format(predictScore))

            loss = np.dot(td, log_td-np.log(pd.reshape(self.outputDim))) / self.outputDim

            cost += loss
            corrects.append(score)
            guesses.append(predictScore)

        for score, item in trees:
            for tree in item:          
                tree.resetFinished()

        return cost/len(trees), pearsonr(corrects,guesses)[0]

    def saveModel(self):
        with open(self.outputFile, "wb") as f:
            cPickle.dump(self.rep_model.stack, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump([self.classifier.hiddenLayer.W_1.eval(), self.classifier.hiddenLayer.W_2.eval(),
                        self.classifier.hiddenLayer.b.eval()], f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump([self.classifier.logRegressionLayer.W.eval(), self.classifier.logRegressionLayer.b.eval()],
                         f, protocol=cPickle.HIGHEST_PROTOCOL)


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
    print "train number %d"%len(trainTrees)
    print "dev number %d"%len(devTrees)
     
    optimizer = Optimization(outputFile=args.outFile, alpha=args.step, optimizer=args.optimizer)

    if args.useLearnedModel=="True":
        startFromEariler = True
    else:
        startFromEariler = False

    optimizer.initial_RepModel(tr, args.repModel, args.wvecDim, startFromEariler=startFromEariler)

    optimizer.initial_theano_mlp(args.hiddenDim, args.outputDim, startFromEariler=startFromEariler)

    best_dev_score  = 0.

    print "training model"
    for e in range(args.epochs):
        
        #print "Running epoch %d"%e
        optimizer.train_with_theano_mlp(trainTrees, args.minibatch)
        #print "Time per epoch : %f"%(end-start)
        cost, dev_score = optimizer.predict(devTrees)
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            optimizer.saveModel()
            print "iter:%d cost: %f dev_score: %f best_dev_score %f"%(e, cost, dev_score, best_dev_score)
        else:
            print "iter:%d cost: %f dev_score: %f"%(e, cost, dev_score)




