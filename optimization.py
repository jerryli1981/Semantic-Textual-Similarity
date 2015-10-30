import numpy as np

from scipy.stats import pearsonr

from rnn import depTreeRnnModel
from lstm import depTreeLSTMModel

from mlp import MLP, my_mlp

import theano.tensor as T
import theano

class Optimization:

    def __init__(self, alpha=0.01, optimizer='sgd'):

        self.learning_rate = alpha # learning rate
        self.optimizer = optimizer
        self.rng = np.random.RandomState(1234)

    def initial_RepModel(self, dep_tree, model, wvecDim, epsilon = 1e-16):

        relNum = len(dep_tree.loadRelMap())

        word2vecs = dep_tree.loadWord2VecMap()

        if model == "RNN":
            rep_model = depTreeRnnModel(relNum, wvecDim)
        elif model == "LSTM":
            rep_model = depTreeLSTMModel(wvecDim)

        rep_model.initialParams(word2vecs, rng=self.rng)

        self.rep_model = rep_model
        self.rep_model.initialGrads()

        if self.optimizer == 'adagrad':
            self.gradt_rnn = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]

        elif self.optimizer =="adadelta":

            self.gradt_rnn_1 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]
            self.gradt_rnn_2 = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]

    def initial_my_mlp(self, hiddenDim, outputDim, epsilon = 1e-16):

        self.classifier = my_mlp(rng=self.rng, n_in=self.rep_model.wvecDim, n_hidden=hiddenDim, n_out=outputDim)

        if self.optimizer == 'adagrad':
            self.gradt_mlp = [epsilon + np.zeros(W.shape) for W in self.classifier.params]

        elif self.optimizer =="adadelta":
            self.gradt_mlp_1 = [epsilon + np.zeros(W.shape) for W in self.classifier.params]
            self.gradt_mlp_2 = [epsilon + np.zeros(W.shape) for W in self.classifier.params]

        self.mlp_forward = self.classifier.predict
        
    def initial_theano_mlp(self, hiddenDim, outputDim, batchMLP=False, epsilon = 1e-16):

        x_1 = T.fvector('x_1')  # the data is presented as one sentence output
        x_2 = T.fvector('x_2')  # the data is presented as one sentence output

        self.batchMLP = batchMLP
        
        L1_reg=0.00
        L2_reg=0.0001 


        if not batchMLP:
            print "Using single mlp"
            y = T.fvector('y')  # the target distribution

            # construct the MLP class
            self.classifier = MLP(rng=self.rng,input_1=x_1, input_2=x_2, n_in=self.rep_model.wvecDim,
                                n_hidden=hiddenDim,n_out=outputDim)

            single_cost = self.classifier.kl_divergence_single(y) + L1_reg * self.classifier.L1+ L2_reg * self.classifier.L2_sqr
            #cost = classifier.kl_divergence(y)
            #cost_function = theano.function([x,y], cost, allow_input_downcast=True)

            hidden_layer_W_1 = self.classifier.hiddenLayer.params[0]
            hidden_layer_W_2 = self.classifier.hiddenLayer.params[1]
            hidden_layer_b = self.classifier.hiddenLayer.params[2]

            deltas_1 = T.dot(hidden_layer_W_1, T.grad(single_cost,hidden_layer_b))
            deltas_2 = T.dot(hidden_layer_W_2, T.grad(single_cost,hidden_layer_b))


            self.deltas_function_1 = theano.function([x_1,x_2,y], deltas_1, allow_input_downcast=True)
            self.deltas_function_2 = theano.function([x_1,x_2,y], deltas_2, allow_input_downcast=True)

            single_gparams = [T.grad(single_cost, param) for param in self.classifier.params]

            self.single_grad = theano.function(inputs=[x_1, x_2, y], outputs=single_gparams,allow_input_downcast=True)

            self.single_cost_function = theano.function(inputs=[x_1, x_2, y], outputs=single_cost ,allow_input_downcast=True)
          
        else:

            print "Using batch mlp"

            x1_batch = T.fmatrix('x1_batch')  # n * d, the data is presented as one sentence output
            x2_batch = T.fmatrix('x2_batch')  # n * d, the data is presented as one sentence output
            y_batch = T.fmatrix('y_batch')  # n * d, the target distribution\

            # construct the MLP class
            self.classifier = MLP(rng=self.rng,input_1=x1_batch, input_2=x2_batch, n_in=self.rep_model.wvecDim,
                                n_hidden=hiddenDim,n_out=outputDim)
            
            batch_cost = self.classifier.kl_divergence_batch(y_batch) + L1_reg * self.classifier.L1 + L2_reg * self.classifier.L2_sqr

            hidden_layer_W_1 = self.classifier.hiddenLayer.params[0]
            hidden_layer_W_2 = self.classifier.hiddenLayer.params[1]
            hidden_layer_b = self.classifier.hiddenLayer.params[2]


            deltas_1 = T.dot(hidden_layer_W_1, T.grad(batch_cost,hidden_layer_b))
            deltas_2 = T.dot(hidden_layer_W_2, T.grad(batch_cost,hidden_layer_b))

            self.deltas_function_1 = theano.function([x1_batch,x2_batch,y_batch], deltas_1, allow_input_downcast=True)
            self.deltas_function_2 = theano.function([x1_batch,x2_batch,y_batch], deltas_2, allow_input_downcast=True)

            batch_gparams = [T.grad(batch_cost, param) for param in self.classifier.params]

            self.batch_grad = theano.function(inputs=[x1_batch, x2_batch, y_batch], 
                outputs=batch_gparams, allow_input_downcast=True)

            self.batch_cost_function = theano.function(inputs=[x1_batch, x2_batch, y_batch], outputs=batch_cost ,allow_input_downcast=True)
          
        
        #for predict
        self.mlp_forward = theano.function([x_1,x_2], self.classifier.predict_p(x_1,x_2), allow_input_downcast=True)

        gparam_hw_1 = T.fmatrix('gparam_hw_1')
        updates_hw_1 = [(self.classifier.params[0], self.classifier.params[0] - self.learning_rate * gparam_hw_1)]

        gparam_hw_2 = T.fmatrix('gparam_hw_2')
        updates_hw_2 = [(self.classifier.params[1], self.classifier.params[1] - self.learning_rate * gparam_hw_2)]

        gparam_hb = T.fvector('gparam_hb')
        updates_hb = [(self.classifier.params[2], self.classifier.params[2] - self.learning_rate * gparam_hb)]

        gparam_lw = T.fmatrix('gparam_lw')
        updates_lw = [(self.classifier.params[3], self.classifier.params[3] - self.learning_rate * gparam_lw)]

        gparam_lb = T.fvector('gparam_lb')
        updates_lb = [(self.classifier.params[4], self.classifier.params[4] - self.learning_rate * gparam_lb)]

        self.update_hw_1_mlp = theano.function(inputs=[gparam_hw_1], updates=updates_hw_1, allow_input_downcast=True)
        self.update_hw_2_mlp = theano.function(inputs=[gparam_hw_2], updates=updates_hw_2, allow_input_downcast=True)
        self.update_hb_mlp = theano.function(inputs=[gparam_hb], updates=updates_hb, allow_input_downcast=True)
        self.update_lw_mlp = theano.function(inputs=[gparam_lw], updates=updates_lw, allow_input_downcast=True)
        self.update_lb_mlp = theano.function(inputs=[gparam_lb], updates=updates_lb, allow_input_downcast=True)


        gparam_hw_1_d = T.fmatrix('gparam_hw_1_d')
        updates_hw_1_d = [(self.classifier.params[0], self.classifier.params[0] + gparam_hw_1_d)]

        gparam_hw_2_d = T.fmatrix('gparam_hw_2_d')
        updates_hw_2_d = [(self.classifier.params[1], self.classifier.params[1] + gparam_hw_2_d)]

        gparam_hb_d = T.fvector('gparam_hb_d')
        updates_hb_d = [(self.classifier.params[2], self.classifier.params[2] + gparam_hb_d)]

        gparam_lw_d = T.fmatrix('gparam_lw_d')
        updates_lw_d = [(self.classifier.params[3], self.classifier.params[3] + gparam_lw_d)]

        gparam_lb_d = T.fvector('gparam_lb_d')
        updates_lb_d = [(self.classifier.params[4], self.classifier.params[4] + gparam_lb_d)]


        self.update_hw_1_mlp_d = theano.function(inputs=[gparam_hw_1_d], updates=updates_hw_1_d, allow_input_downcast=True)
        self.update_hw_2_mlp_d = theano.function(inputs=[gparam_hw_2_d], updates=updates_hw_2_d, allow_input_downcast=True)
        self.update_hb_mlp_d = theano.function(inputs=[gparam_hb_d], updates=updates_hb_d, allow_input_downcast=True)
        self.update_lw_mlp_d = theano.function(inputs=[gparam_lw_d], updates=updates_lw_d, allow_input_downcast=True)
        self.update_lb_mlp_d = theano.function(inputs=[gparam_lb_d], updates=updates_lb_d, allow_input_downcast=True)


        if self.optimizer == 'adagrad':
            self.gradt_mlp = [epsilon + np.zeros(W.shape.eval()) for W in self.classifier.params]

        elif self.optimizer =="adadelta":
            self.gradt_mlp_1 = [epsilon + np.zeros(W.shape.eval()) for W in self.classifier.params]
            self.gradt_mlp_2 = [epsilon + np.zeros(W.shape.eval()) for W in self.classifier.params]

    def train_with_my_mlp(self, trainData, batchSize):
        np.random.shuffle(trainData)

        batches = [trainData[idx : idx + batchSize] for idx in xrange(0, len(trainData), batchSize)]

        for index, batchData in enumerate(batches):

            cost, mlp_grad = self.costAndGrad_mymlp_single_grad(batchData)

            for score, item in batchData:
                for tree in item:          
                    tree.resetFinished()

            if self.optimizer == 'adagrad':

                self.gradt_mlp = [gt+g**2 for gt,g in zip(self.gradt_mlp, mlp_grad)]

                dparam =  [g*(1./np.sqrt(gt)) for gt,g in zip(self.gradt_mlp, mlp_grad)]

                self.classifier.params = [P - self.learning_rate*dP for P,dP in zip(self.classifier.params, dparam)]

                self.adagrad_rnn(self.rep_model.dstack)

            elif self.optimizer == 'adadelta':
                #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
                self.gradt_mlp_1 = [rho*gt+(1.0-rho)*(g**2) for gt,g in zip(self.gradt_mlp_1, grad)]
 
                #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
                dparam = [ -(np.sqrt(gt2+eps) / np.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(self.gradt_mlp_1, self.gradt_mlp_2, grad)]
                self.classifier.params = [P + self.learning_rate*dP for P,dP in zip(self.classifier.params, dparam)]
                
                #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
                self.gradt_mlp_2 = [rho*dt + (1.0-rho)*(d ** 2) for dt, d in zip(self.gradt_mlp_2, dparam)]


                self.adadelta_rnn(self.rep_model.dstack)


    def train_with_theano_mlp(self, trainData, batchSize):

        np.random.shuffle(trainData)

        batches = [trainData[idx : idx + batchSize] for idx in xrange(0, len(trainData), batchSize)]

        for index, batchData in enumerate(batches):

            if self.batchMLP:
                cost, mlp_grad = self.costAndGrad_theano_batch_grad(batchData)
            else:
                cost, mlp_grad = self.costAndGrad_theano_single_grad(batchData)
            
            for score, item in batchData:
                for tree in item:          
                    tree.resetFinished()

            #begin to update rnn parameters
            if self.optimizer == 'sgd':

                #begin to update mlp parameters
                self.update_hw_1_mlp(mlp_grad[0])
                self.update_hw_2_mlp(mlp_grad[1])
                self.update_hb_mlp(mlp_grad[2])
                self.update_lw_mlp(mlp_grad[3])
                self.update_lb_mlp(mlp_grad[4])

                #begin to update rnn parameters
                update = self.rep_model.dstack

                self.rep_model.stack[1:] = [P-self.learning_rate*dP for P,dP in zip(self.rep_model.stack[1:],update[1:])]

                # handle dictionary update sparsely
                dL = update[0]
                for j in range(self.rep_model.numWords):
                    self.rep_model.L[:,j] -= self.learning_rate*dL[:,j]

            elif self.optimizer == 'adagrad':

                self.adagrad_mlp(mlp_grad)

                self.adagrad_rnn(self.rep_model.dstack)

            elif self.optimizer == 'adadelta':

                self.adadelta_mlp(mlp_grad)

                self.adadelta_rnn(self.rep_model.dstack)

    def costAndGrad_theano_single_grad(self, trainData, epsilon = 1e-16):

        targetData = np.zeros((len(trainData), self.classifier.numLabels+1))

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

        d_hidden_w_1 = np.zeros((self.classifier.hiddenLayer.params[0].shape.eval()))
        d_hidden_w_2 = np.zeros((self.classifier.hiddenLayer.params[1].shape.eval()))
        d_hidden_b = np.zeros((self.classifier.hiddenLayer.params[2].shape.eval()))
        d_log_w = np.zeros((self.classifier.logRegressionLayer.params[0].shape.eval()))
        d_log_b = np.zeros((self.classifier.logRegressionLayer.params[1].shape.eval()))

        mlp_grad = [d_hidden_w_1, d_hidden_w_2, d_hidden_b, d_log_w, d_log_b]

        cost = 0.

        for i, (score, item) in enumerate(trainData): 

            td = targetData[i]
            td += epsilon

            first_tree_rep = self.rep_model.forwardProp(item[0])
            second_tree_rep = self.rep_model.forwardProp(item[1])
            mul_rep = first_tree_rep * second_tree_rep
            sub_rep = np.abs(first_tree_rep-second_tree_rep)
    
            deltas_1 = self.deltas_function_1(mul_rep, sub_rep, td)
            deltas_2 = self.deltas_function_2(mul_rep, sub_rep, td)

            self.rep_model.backProp(item[0], deltas_1)
            #self.rep_model.backProp(item[0], deltas_2)
            #self.rep_model.backProp(item[1], deltas_1)
            self.rep_model.backProp(item[1], deltas_2)
            
            [dhw_1, dhw_2, dhb, dlw, dlb] = self.single_grad(mul_rep, sub_rep, td)
            d_hidden_w_1 += dhw_1
            d_hidden_w_2 += dhw_2
            d_hidden_b += dhb
            d_log_w += dlw
            d_log_b += dlb

            cost += self.single_cost_function(mul_rep, sub_rep, td)

        cost = cost / len(trainData)

        return cost , mlp_grad

    def costAndGrad_theano_batch_grad(self, trainData, epsilon = 1e-16):

        targetData = np.zeros((len(trainData), self.classifier.numLabels+1))

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

        cost = 0.

        mul_reps = np.zeros((len(trainData), self.rep_model.wvecDim))
        sub_reps = np.zeros((len(trainData), self.rep_model.wvecDim))

        for i, (score, item) in enumerate(trainData): 

            td = targetData[i]
            td += epsilon

            first_tree_rep = self.rep_model.forwardProp(item[0])
            second_tree_rep = self.rep_model.forwardProp(item[1])
            mul_rep = first_tree_rep * second_tree_rep
            sub_rep = np.abs(first_tree_rep-second_tree_rep)

            mul_reps[i] = mul_rep
            sub_reps[i] = sub_rep
    
 
        deltas_1 = self.deltas_function_1(mul_reps, sub_reps, targetData)  
        deltas_2 = self.deltas_function_2(mul_reps, sub_reps, targetData) 

        """
        here maybe just update one tree
        """ 
        deltas_1 /= len(trainData)
        deltas_2 /= len(trainData)

        for i, (score, item) in enumerate(trainData): 

            self.rep_model.backProp(item[0], deltas_1)
            self.rep_model.backProp(item[1], deltas_2)


        cost = self.batch_cost_function(mul_reps, sub_reps, targetData)

        cost /= len(trainData)

        mlp_grad = self.batch_grad(mul_reps, sub_reps, targetData)

        return cost , mlp_grad

    def costAndGrad_mymlp_single_grad(self, trainData, epsilon = 1e-16):

        targetData = np.zeros((len(trainData), self.classifier.numLabels+1))

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

        cost = 0.

        for i, (score, item) in enumerate(trainData): 

            td = targetData[i]
            td += epsilon

            first_tree_rep = self.rep_model.forwardProp(item[0])
            second_tree_rep = self.rep_model.forwardProp(item[1])
            mul_rep = first_tree_rep * second_tree_rep
            sub_rep = np.abs(first_tree_rep-second_tree_rep)

            c, activation, pd = self.classifier.forwardProp(mul_rep, sub_rep, td)
    
            deltas_1, deltas_2 = self.classifier.backwardProp(mul_rep, sub_rep, activation, pd, td)

            self.rep_model.backProp(item[0], deltas_1)
            self.rep_model.backProp(item[1], deltas_2)

            cost += c

        cost /= len(trainData)

        return cost , self.classifier.dstack

        
    def adagrad_mlp(self, grad):

        self.gradt_mlp = [gt+g**2 for gt,g in zip(self.gradt_mlp, grad)]

        update =  [g*(1./np.sqrt(gt)) for gt,g in zip(self.gradt_mlp,grad)]

        self.update_hw_1_mlp(update[0])
        self.update_hw_2_mlp(update[1])
        self.update_hb_mlp(update[2])
        self.update_lw_mlp(update[3])
        self.update_lb_mlp(update[4])

    def adadelta_mlp(self, grad, eps=1e-6, rho=0.95):

        #param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2) 
        self.gradt_mlp_1 = [rho*gt+(1.0-rho)*(g**2) for gt,g in zip(self.gradt_mlp_1, grad)]
 
        #dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
        dparam = [ -(np.sqrt(gt2+eps) / np.sqrt(gt1+eps) ) * g for gt1, gt2, g in zip(self.gradt_mlp_1, self.gradt_mlp_2, grad)]

        self.update_hw_1_mlp_d(dparam[0])
        self.update_hw_2_mlp_d(dparam[1])
        self.update_hb_mlp_d(dparam[2])
        self.update_lw_mlp_d(dparam[3])
        self.update_lb_mlp_d(dparam[4])

        #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
        self.gradt_mlp_2 = [rho*dt + (1.0-rho)*(d ** 2) for dt, d in zip(self.gradt_mlp_2, dparam)]

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
            dLt[:,j] = dLt[:,j] + dL[:,j]**2
            dL[:,j] = dL[:,j] * (1./np.sqrt(dLt[:,j]))

        # handle dictionary update sparsely
        for j in range(self.rep_model.numWords):
            self.rep_model.L[:,j] -= self.learning_rate*dL[:,j]

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
            dLt_1[:,j] = rho*dLt_1[:,j]+(1.0-rho)*(dL[:,j] ** 2)
            dL[:,j] = -( np.sqrt(dLt_2[:,j]+eps)/ np.sqrt(dLt_1[:,j]+eps) ) * dL[:,j]

            #update
            dLt_2[:,j] = rho*dLt_2[:,j] + (1.0-rho)*(dL[:,j] ** 2)

        for j in range(self.rep_model.numWords):
            self.rep_model.L[:,j] += dL[:,j]

        #updates.append((param_update_2, rho*param_update_2+(1. - rho)*(dparam ** 2)))
        self.gradt_rnn_2[1:] = [rho*dt + (1.0-rho)*( d** 2) for dt, d in zip(self.gradt_rnn_2[1:], dparam)]

    def predict(self, trees, epsilon=1e-16):

        targets = np.zeros((len(trees), self.classifier.numLabels+1))
  
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
            pd = self.mlp_forward(mul_rep,sub_rep)

            predictScore = pd.reshape(self.classifier.numLabels).dot(np.array([1,2,3,4,5]))

            predictScore = float("{0:.2f}".format(predictScore))

            loss = np.dot(td, log_td-np.log(pd.reshape(self.classifier.numLabels))) / self.classifier.numLabels

            cost += loss
            corrects.append(score)
            guesses.append(predictScore)

        for score, item in trees:
            for tree in item:          
                tree.resetFinished()

        return cost/len(trees), pearsonr(corrects,guesses)[0]

