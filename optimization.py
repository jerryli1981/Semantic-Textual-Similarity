import numpy as np
from multiprocessing import Pool


from scipy.stats import pearsonr

from rnn import depTreeRnnModel
from lstm import depTreeLSTMModel

from mlp import MLP

import theano.tensor as T
import theano


def roll_params(params):
    L, WR, WV, b, Wsg, Wsm, bsm = params
    return np.concatenate((L.ravel(), WR.ravel(), WV.ravel(), b.ravel(), Wsg.ravel(), Wsm.ravel(), bsm.ravel()))

def unroll_params(arr, hparams):

    relNum, wvecDim, outputDim, numWords, sim_nhidden = hparams

    ind = 0

    d = wvecDim*wvecDim

    L = arr[ind : ind + numWords*wvecDim].reshape( (wvecDim, numWords) )
    ind +=numWords*wvecDim

    WR = arr[ind : ind + relNum*d].reshape( (relNum, wvecDim, wvecDim) )
    ind += relNum*d

    WV = arr[ind : ind + d].reshape( (wvecDim, wvecDim) )
    ind += d

    b = arr[ind : ind + wvecDim].reshape(wvecDim,)
    ind += wvecDim

    Wsg = arr[ind : ind + sim_nhidden * wvecDim].reshape((sim_nhidden, wvecDim))
    ind += sim_nhidden * wvecDim

    Wsm = arr[ind : ind + outputDim*sim_nhidden].reshape( (outputDim, sim_nhidden))
    ind += outputDim*sim_nhidden

    bsm = arr[ind : ind + outputDim].reshape(outputDim,)

    return (L, WR, WV, b, Wsg, Wsm, bsm)

def unwrap_self_forwardBackwardProp(arg, **kwarg):
    return depTreeRnnModel.forwardBackwardProp(*arg, **kwarg)


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

    def initial_theano_mlp(self, outputDim, epsilon = 1e-16):

        x_1 = T.fvector('x_1')  # the data is presented as one sentence output
        x_2 = T.fvector('x_2')  # the data is presented as one sentence output
        y = T.fvector('y')  # the target distribution

        # construct the MLP class
        self.classifier = MLP(rng=self.rng,input_1=x_1, input_2=x_2, n_in=self.rep_model.wvecDim,
                            n_hidden=50,n_out=outputDim)

   
        self.mlp_forward = theano.function([x_1,x_2], self.classifier.predict_p(x_1,x_2), allow_input_downcast=True)
        
        L1_reg=0.00
        L2_reg=0.0001 

        cost = self.classifier.kl_divergence(y) + L1_reg * self.classifier.L1+ L2_reg * self.classifier.L2_sqr
        #cost = classifier.kl_divergence(y)
        #cost_function = theano.function([x,y], cost, allow_input_downcast=True)

        hidden_layer_W_1 = self.classifier.hiddenLayer.params[0]
        hidden_layer_W_2 = self.classifier.hiddenLayer.params[1]
        hidden_layer_b = self.classifier.hiddenLayer.params[2]

        deltas_1 = T.dot(hidden_layer_W_1, T.grad(cost,hidden_layer_b))
        deltas_2 = T.dot(hidden_layer_W_2, T.grad(cost,hidden_layer_b))

        """
        from rnn_mlp_chunk

        x = T.fvector("x")
        y = T.fvector("y")

        W_hidden, b_hidden = self.classifier.hiddenLayer.params

        W_lg, b_lg = self.classifier.logRegressionLayer.params

        output = T.nnet.sigmoid(T.dot(x, W_hidden) + b_hidden)

        p_y_given_x = T.nnet.softmax(T.dot(output, W_lg) + b_lg)

        newshape=(T.shape(p_y_given_x)[1],)
 
        kl_divergence = T.dot(y, (T.log(y)-T.log(T.reshape(p_y_given_x, newshape))).T) / self.rep_model.outputDim

        deltas = T.dot(W_hidden, T.grad(kl_divergence, b_hidden))

        self.deltas_function = theano.function([x, y], deltas, allow_input_downcast=True)

        """


        #grad_function = theano.function([x,y], T.grad(cost,hidden_layer_b), allow_input_downcast=True)
        self.deltas_function_1 = theano.function([x_1,x_2,y], deltas_1, allow_input_downcast=True)
        self.deltas_function_2 = theano.function([x_1,x_2,y], deltas_2, allow_input_downcast=True)

        gparams = [T.grad(cost, param) for param in self.classifier.params]

        self.accu_grad = theano.function(inputs=[x_1, x_2, y], outputs=gparams,allow_input_downcast=True)

        x1_batch = T.fmatrix('x1_batch')  # n * d, the data is presented as one sentence output
        x2_batch = T.fmatrix('x2_batch')  # n * d, the data is presented as one sentence output
        y_batch = T.fmatrix('y_batch')  # n * d, the target distribution
        self.batch_accu_grad = theano.function(inputs=[x1_batch, x2_batch, y_batch], 
            outputs=gparams,allow_input_downcast=True)


        """
        updates = [ (param, param - self.learning_rate * gparam)
                    for param, gparam in zip(self.classifier.params, gparams)
                  ]

        #self.update_params_mlp = theano.function(inputs=[x, y], outputs=cost, updates=updates,allow_input_downcast=True)
        """
        #for gradient check
        self.cost_function = theano.function(inputs=[x_1, x_2, y], outputs=cost ,allow_input_downcast=True)
      
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

    def train(self, trainData, batchSize):

        np.random.shuffle(trainData)

        batches = [trainData[idx : idx + batchSize] for idx in xrange(0, len(trainData), batchSize)]

        for index, batchData in enumerate(batches):

            cost, mlp_grad = self.costAndGrad_theano_single_grad(batchData)
            #cost, mlp_grad = self.costAndGrad_theano_batch_grad(batchData)

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
            
            [dhw_1, dhw_2, dhb, dlw, dlb] = self.accu_grad(mul_rep, sub_rep, td)
            d_hidden_w_1 += dhw_1
            d_hidden_w_2 += dhw_2
            d_hidden_b += dhb
            d_log_w += dlw
            d_log_b += dlb

            cost += self.cost_function(mul_rep, sub_rep, td)

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
    
            deltas_1 = self.deltas_function_1(mul_rep, sub_rep, td)
            deltas_2 = self.deltas_function_2(mul_rep, sub_rep, td)

            self.rep_model.backProp(item[0], deltas_1)
            self.rep_model.backProp(item[1], deltas_2)
            
            cost += self.cost_function(mul_rep, sub_rep, td)

        cost = cost / len(trainData)

        mlp_grad = self.batch_accu_grad(mul_reps, sub_reps, targetData)

        return cost , mlp_grad

    def costAndGrad(self, batchData, rho=1e-4, numProc= None):
        raise "current not work"

        def forwardBackwardProp(self, mbdata):
            cost = 0.0
            for tree in mbdata: 
                cost += self.forwardProp(tree)
                self.backProp(tree)

            return cost, roll_params((self.dL, self.dWR, self.dWV, self.db, self.dWsg, self.dWsm, self.dbsm))

        if numProc == None:
            cost, grad = self.forwardBackwardProp(mbdata)
        else:

            miniBatchSize = len(batchData) / numProc

            pool = Pool(processes = numProc)
            
            miniBatchData = [batchData[i:i+miniBatchSize] for i in range(0, len(batchData), miniBatchSize)]

            result = pool.map(unwrap_self_forwardBackwardProp, zip([self]*len(miniBatchData), miniBatchData))

            pool.close() #no more processed accepted by this pool
            pool.join() #wait until all processes are finished

            cost = 0.
            grad = None
            for mini_cost, mini_grads in result:
                cost += mini_cost
                if grad is None:
                    grad = mini_grads
                else:
                    grad += mini_grads

        hparams = (self.relNum, self.wvecDim, self.classifier.numLabels, self.numWords)

        self.dL, self.dWR, self.dWV, self.db, self.dWs, self.dbs = unroll_params(grad, hparams)

        # scale cost and grad by mb size
        scale = (1./len(batchData))
        for v in range(self.numWords):
            self.dL[:,v] *= scale
            
        # Add L2 Regularization 
        cost += (rho/2)*np.sum(self.WV**2)
        cost += (rho/2)*np.sum(self.WR**2)
        cost += (rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dWR + rho*self.WR),
                                   scale*(self.dWV + rho*self.WV),
                                   scale*self.db,
                                   scale*(self.dWs+rho*self.Ws),scale*self.dbs]


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

