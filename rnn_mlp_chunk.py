import numpy as np
import random # for random shuffle train data
from numpy import inf

import theano
import theano.tensor as T

from scipy.stats import pearsonr


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]

        self.input = input

#T.tanh
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):

        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.numLabels = n_out

        self.input = input

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=self.input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.sigmoid
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.output = self.logRegressionLayer.p_y_given_x

    def kl_divergence(self, y):

        return T.sum(y * (T.log(y) - T.log(self.output))) / (self.numLabels * y.shape[0])


    def predict_p(self, new_data):

        W_hidden, b_hidden = self.hiddenLayer.params

        W_lg, b_lg = self.logRegressionLayer.params

        output = T.nnet.sigmoid(T.dot(new_data, W_hidden) + b_hidden)

        p_y_given_x = T.nnet.softmax(T.dot(output, W_lg) + b_lg)

        return p_y_given_x
   

class depTreeRnnModel:

    def __init__(self, relNum, wvecDim, outputDim):

        self.relNum = relNum
        self.wvecDim = wvecDim
        self.outputDim = outputDim


    def initialParams(self, word2vecs, rng):

        # Word vectors
        self.numWords = word2vecs.shape[1]
        self.L = word2vecs[:self.wvecDim, :]

        # Relation layer parameters
        self.WR = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.relNum, self.wvecDim, self.wvecDim)
                    )

        # Hidden layer parameters 
        self.WV = rng.uniform(
                    low=-np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    high=np.sqrt(6. / (self.wvecDim + self.wvecDim)),
                    size=(self.wvecDim, self.wvecDim)
                    )

        self.b = np.zeros((self.wvecDim))

        self.stack = [self.L, self.WR, self.WV, self.b]

    def initialGrads(self):

        #create with default_factory : defaultVec = lambda : np.zeros((wvecDim,))
        #make the defaultdict useful for building a dictionary of np array
        #this is very good to save memory, However, this may not support for multiprocess due to pickle error

        #defaultVec = lambda : np.zeros((wvecDim,))
        #dL = collections.defaultdict(defaultVec)
        self.dL = np.zeros((self.wvecDim, self.numWords))
        self.dWR = np.zeros((self.relNum, self.wvecDim, self.wvecDim))
        self.dWV = np.zeros((self.wvecDim, self.wvecDim))
        self.db = np.zeros(self.wvecDim)

        self.dstack = [self.dL, self.dWR, self.dWV, self.db]
        

    def forwardProp(self, tree):

        #because many training epoch. 
        tree.resetFinished()

        to_do = []
        to_do.append(tree.root)

        while to_do:
        
            curr = to_do.pop(0)
            curr.vec = self.L[:, curr.index]

            # node is leaf
            if len(curr.kids) == 0:
                # activation function is the normalized tanh
                curr.hAct= np.tanh(np.dot(self.WV,curr.vec) + self.b)
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
                    sum = np.zeros((self.wvecDim))
                    for i, rel in curr.kids:
                        pass
                        rel_vec = self.WR[rel.index]
                        sum += rel_vec.dot(tree.nodes[i].hAct) 

                    curr.hAct = np.tanh(sum + self.WV.dot(curr.vec) + self.b)
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        return tree.root.hAct


    def backProp(self, tree, deltas):

        to_do = []
        to_do.append(tree.root)

        tree.root.deltas = deltas

        while to_do:

            curr = to_do.pop(0)

            if len(curr.kids) == 0:

                self.dL[:, curr.index] += curr.deltas

            else:

                # derivative of tanh
                curr.deltas *= (1-curr.hAct**2)

                self.dWV += np.outer(curr.deltas, curr.vec)
                self.db += curr.deltas

                for i, rel in curr.kids:

                    kid = tree.nodes[i]
                    to_do.append(kid)

                    self.dWR[rel.index] += np.outer(curr.deltas, kid.hAct)

                    rel_vec = self.WR[rel.index]
                    kid.deltas = np.dot(rel_vec.T, curr.deltas)

    def updateParams(self,learning_rate,update):
        """
        Updates parameters as
        p := p - learning_rate * update.
        """
        self.stack[1:] = [P-learning_rate*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in range(self.numWords):
            self.L[:,j] -= learning_rate*dL[:,j]

class SGD:

    def __init__(self, trainData, batchSize, rep_model, rng, merged=False, 
        alpha=0.01, optimizer='sgd', epsilon = 1e-16):

        self.trainData = trainData
        self.batchSize = batchSize
        self.rep_model = rep_model
        self.learning_rate = alpha # learning rate

        x_batch = T.fmatrix('x_batch')  # n * d, the data is presented as one sentence output
        y_batch = T.fmatrix('y_batch')  # n * d, the target distribution

        # construct the MLP class
        if merged:
            mlp = MLP(rng=rng,input=x_batch,n_in=self.rep_model.wvecDim,n_hidden=150,n_out=self.rep_model.outputDim)
        else:
            mlp = MLP(rng=rng,input=x_batch,n_in=self.rep_model.wvecDim*2,n_hidden=150,n_out=self.rep_model.outputDim)

        self.classifier = mlp 

        L1_reg=0.00
        L2_reg=0.0001

        self.cost = self.classifier.kl_divergence(y_batch) + L1_reg * self.classifier.L1+ L2_reg * self.classifier.L2_sqr
   
        
        gparams = [T.grad(self.cost, param) for param in self.classifier.params]

        self.mlp_updates = [ (param, param - self.learning_rate * gparam)
                    for param, gparam in zip(self.classifier.params, gparams)]

        self.train_mlp_model = theano.function(inputs=[x_batch, y_batch], 
                                outputs=self.cost, 
                                updates=self.mlp_updates,
                                allow_input_downcast=True)

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

        self.rep_model.initialGrads()

        self.optimizer = optimizer

        if self.optimizer == 'sgd':
            print "Using sgd.."

        elif self.optimizer == 'adagrad':
            print "Using adagrad..."
            self.gradt = [epsilon + np.zeros(W.shape) for W in self.rep_model.stack]

    def run(self, epsilon = 1e-16):

        random.shuffle(self.trainData)

        batches = [self.trainData[idx : idx + self.batchSize] for idx in xrange(0, len(self.trainData), self.batchSize)]

        targetData = np.zeros((len(self.trainData), self.rep_model.outputDim+1))

        for i, (score, item) in enumerate(self.trainData):
            sim = score
            ceil = np.ceil(sim)
            floor = np.floor(sim)
            if ceil == floor:
                targetData[i, floor] = 1
            else:
                targetData[i, floor] = ceil-sim 
                targetData[i, ceil] = sim-floor

        targetData = targetData[:, 1:]

        for index, batchData in enumerate(batches):

            targets = targetData[index * self.batchSize: (index + 1) * self.batchSize]

            tree_reps = np.zeros((len(targets), self.rep_model.wvecDim*2))
            
            for i, (score, item) in enumerate(batchData): 

                td = targets[i]
                td += epsilon

                if len(item) ==1:

                    merged_tree_rep = self.rep_model.forwardProp(item[0])

                elif len(item) == 2:
                    first_tree_rep = self.rep_model.forwardProp(item[0])
                    second_tree_rep = self.rep_model.forwardProp(item[1])
                    merged_tree_rep = np.concatenate((first_tree_rep, second_tree_rep))

                tree_reps[i] = merged_tree_rep

                deltas = self.deltas_function(merged_tree_rep, td) # (n_hidden,)

                if len(item) == 1:
                    self.rep_model.backProp(item[0], deltas)
                elif len(item) == 2:
                    self.rep_model.backProp(item[0], deltas[:self.rep_model.wvecDim])
                    self.rep_model.backProp(item[1], deltas[self.rep_model.wvecDim:])

                
            minibatch_avg_cost = self.train_mlp_model(tree_reps, targets)

            #loss += 0.5*(np.sum(self.WV**2) + np.sum(self.WR**2) + optimizer.classifier.L2_sqr)

            for score, item in batchData:
                for tree in item:          
                    tree.resetFinished()

            #begin to update rnn parameters
            grad = self.rep_model.dstack

            if self.optimizer == 'sgd':

                update = grad
                scale = -self.learning_rate

            elif self.optimizer == 'adagrad':
                # trace = trace+grad.^2
                self.gradt[1:] = [gt+g**2 
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # update = grad.*trace.^(-1/2)
                update =  [g*(1./np.sqrt(gt))
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # handle dictionary separately
                dL = grad[0]
                dLt = self.gradt[0]
                for j in range(self.rep_model.numWords):
                    dLt[:,j] = dLt[:,j] + dL[:,j]**2
                    dL[:,j] = dL[:,j] * (1./np.sqrt(dLt[:,j]))
                update = [dL] + update
                scale = -self.learning_rate

            # update params
            self.rep_model.updateParams(scale,update)

            return self.rep_model, self.classifier

def predict(trees, rnn, classifier, epsilon=1e-16):

    # get target distribution for batch
    targets = np.zeros((len(trees), rnn.outputDim+1))
    # compute target distribution 

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

    x = T.fvector('x')  # the data is presented as one sentence output
   
    mlp_forward = theano.function([x], classifier.predict_p(x), allow_input_downcast=True)

    cost = 0
    corrects = []
    guesses = []

    for i, (score, item) in enumerate(trees):

        td = targets[i]
        td += epsilon
        log_td = np.log(td)

        if len(item) == 1:
            tree_rep= rnn.forwardProp(item[0])
            pd = mlp_forward(tree_rep)

        elif len(item) == 2:
            first_tree_rep= rnn.forwardProp(item[0])
            second_tree_rep = rnn.forwardProp(item[1])
            merged_tree_rep = np.concatenate((first_tree_rep, second_tree_rep))
            pd = mlp_forward(merged_tree_rep)

        predictScore = pd.reshape(rnn.outputDim).dot(np.array([1,2,3,4,5]))

        predictScore = float("{0:.2f}".format(predictScore))

        loss = np.dot(td, log_td-np.log(pd.reshape(rnn.outputDim))) / rnn.outputDim

        cost += loss
        corrects.append(score)
        guesses.append(predictScore)

    #print corrects[:10]
    #print guesses[:10]

    for score, item in trees:
        for tree in item:          
            tree.resetFinished()

    return cost/len(trees), pearsonr(corrects,guesses)[0]

if __name__ == '__main__':

    import dependency_tree as tr     
    merged = False
    trainTrees = tr.loadTrees("train", merged=merged)
    print "train number %d"%len(trainTrees)
     
    numW = len(tr.loadWordMap())

    relMap = tr.loadRelMap()
    relNum = len(relMap)
    word2vecs = tr.loadWord2VecMap()

    wvecDim = 100
    outputDim = 5

    rng = np.random.RandomState(1234)

    rnn = depTreeRnnModel(relNum, wvecDim, outputDim)

    rnn.initialParams(word2vecs, rng=rng)
    minibatch = 200

    optimizer = SGD(trainTrees, minibatch, rep_model=rnn, rng=rng, merged=merged, optimizer='sgd')

    devTrees = tr.loadTrees("dev", merged=merged)

    best_dev_score  = 0.

    print "training model"
    for e in range(100):
        
        #print "Running epoch %d"%e
        rnn, mlp = optimizer.run()
        #print "Time per epoch : %f"%(end-start)
        cost, dev_score = predict(devTrees, rnn, mlp)
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print "iter:%d cost: %f dev_score: %f best_dev_score %f"%(e, cost, dev_score, best_dev_score)
        else:
            print "iter:%d cost: %f dev_score: %f"%(e, cost, dev_score) 
        
        

    

    

