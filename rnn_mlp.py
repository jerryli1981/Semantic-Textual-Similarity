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

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

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
            activation=T.tanh
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
        newshape=(T.shape(self.output)[1],)
        x = T.reshape(self.output, newshape)
        return T.dot(y, (T.log(y) - T.log(x)).T) / self.numLabels


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


    def initialParams(self, word2vecs):

        #generate the same random number
        np.random.seed(12341)

        r = np.sqrt(6)/np.sqrt(201)

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.numWords = word2vecs.shape[1]
        # scale by 0.01 can pass gradient check
        self.L = word2vecs[:self.wvecDim, :]

        # Relation layer parameters
        #self.WR = 0.01*np.random.randn(self.relNum, self.wvecDim, self.wvecDim)
        self.WR = np.random.rand(self.relNum, self.wvecDim, self.wvecDim) * 2 * r -r

        # Hidden layer parameters 
        #self.WV = 0.01*np.random.randn(self.wvecDim, self.wvecDim)
        self.WV = np.random.rand(self.wvecDim, self.wvecDim) * 2 * r - r
        self.b = np.zeros((self.wvecDim))

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

        cost  =  0.0 

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


    #Minbatch stochastic gradient descent
    def train(self, trainData, batchSize):

        #random.shuffle(trainData)

        batches = [trainData[x : x + batchSize] for x in xrange(0, len(trainData), batchSize)]

        x = T.fvector('x')  # the data is presented as one sentence output
        y = T.fvector('y')  # the target distribution

        rng = np.random.RandomState(1234)

        self.initialGrads()

        # construct the MLP class
        classifier = MLP(
            rng=rng,
            input=x,
            n_in=self.wvecDim,
            n_hidden=150,
            n_out=self.outputDim
        )

        epsilon = 1e-8
        L1_reg=0.00
        L2_reg=0.0001

        #cost_function = theano.function([x,y], cost, allow_input_downcast=True)

        cost = classifier.kl_divergence(y) + L1_reg * classifier.L1+ L2_reg * classifier.L2_sqr
        #cost = classifier.kl_divergence(y)

        hidden_layer_W = classifier.hiddenLayer.params[0]
        hidden_layer_b = classifier.hiddenLayer.params[1]
        deltas = T.dot(hidden_layer_W, T.grad(cost,hidden_layer_b))

        #grad_function = theano.function([x,y], T.grad(cost,hidden_layer_b), allow_input_downcast=True)
        deltas_function = theano.function([x,y], deltas, allow_input_downcast=True)

        learning_rate=0.01

        gparams = [T.grad(cost, param) for param in classifier.params]

        updates = [ (param, param - learning_rate * gparam)
                    for param, gparam in zip(classifier.params, gparams)
                  ]

        cost_function = theano.function(inputs=[x, y], outputs=cost, updates=updates,allow_input_downcast=True)

        for batchData in batches:

            # get target distribution for batch
            targets = np.zeros((batchSize, self.outputDim+1))
            # compute target distribution 
    
            for i, tree in enumerate(batchData):
                sim = tree.score
                ceil = np.ceil(sim)
                floor = np.floor(sim)
                if ceil == floor:
                    targets[i, floor] = 1
                else:
                    targets[i, floor] = ceil-sim
                    targets[i, ceil] = sim-floor

            targets = targets[:, 1:]

            loss = 0.

            for i, tree in enumerate(batchData): 

                tree_rep = self.forwardProp(tree)

                """
                norm = -1.0/self.outputDim
                sim_grad = norm * tree.root.td
                sim_grad[sim_grad == -0.] = 0

                deltas_sm = sim_grad

                deltas_sg = np.dot(Wsm.T, deltas_sm)
  
                deltas_sg *= tree.root.sigAct*(1-tree.root.sigAct)
              
                deltas = np.dot(self.Wsg.T,deltas_sg)
                """

                td = targets[i]
                td += epsilon


                #due to hidden_layer_b_grad equal delta up, so based on error propogation
                deltas = deltas_function(tree_rep,td) # (n_hidden,)

                self.backProp(tree, deltas)
                
                """
                output = mlp_forward(tree_rep)
                log_td = np.log(td)

                example_loss = np.dot(td, log_td-np.log(output.reshape(self.outputDim))) / self.outputDim
                """

                example_loss = cost_function(tree_rep,td)
                
                #assert np.abs(example_loss-example_loss_2)< 0.00001, "Shit"

                """
                norm = -1.0/self.outputDim
                sim_grad = norm * td
                sim_grad[sim_grad == -0.] = 0
                """

                loss += example_loss

            loss = loss / batchSize

            loss += 0.5*(np.sum(self.WV**2) + np.sum(self.WR**2) + classifier.L2_sqr)

            #begin to update parameters

            stack = [self.L, self.WR, self.WV, self.b]

            gradt = [epsilon + np.zeros(W.shape) for W in stack]

            grad = self.dstack

            # trace = trace+grad.^2
            gradt[1:] = [gt+g**2 for gt,g in zip(gradt[1:],grad[1:])]

            # update = grad.*trace.^(-1/2)
            update =  [g*(1./np.sqrt(gt)) for gt,g in zip(gradt[1:],grad[1:])]

            # handle dictionary separately
            dL = grad[0]
            dLt = gradt[0]
            for j in range(self.numWords):
                dLt[:,j] = dLt[:,j] + dL[:,j]**2
                dL[:,j] = dL[:,j] * (1./np.sqrt(dLt[:,j]))

            alpha = 0.01

            scale = -alpha

            update = [dL] + update

            stack[1:] = [P+scale*dP for P,dP in zip(stack[1:],update[1:])]

            for j in range(self.numWords):
                self.L[:,j] += scale*dL[:,j]

        for tree in trainData:
                tree.resetFinished()

        return classifier

    def predict(self, trees, classifier, epsilon = 1e-8):

        # get target distribution for batch
        targets = np.zeros((len(trees), self.outputDim+1))
        # compute target distribution 

        for i, tree in enumerate(trees):
            sim = tree.score
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
        for i, tree in enumerate(trees):

            td = targets[i]
            td += epsilon

            log_td = np.log(td)

            tree_rep= self.forwardProp(tree)

            pd = mlp_forward(tree_rep)

            predictScore = pd.reshape(self.outputDim).dot(np.array([1,2,3,4,5]))

            predictScore = float("{0:.2f}".format(predictScore))

            loss = np.dot(td, log_td-np.log(pd.reshape(self.outputDim))) / self.outputDim

            cost += loss
            corrects.append(tree.score)
            guesses.append(predictScore)

        print corrects[:10]
        print guesses[:10]
        print "Cost %f"%(cost/len(trees))    
        #print "Pearson correlation %f"%(pearsonr(corrects,guesses)[0])

        for tree in trees:
                tree.resetFinished()

        #print "Spearman correlation %f"%(spearmanr(corrects,guesses)[0])
        return pearsonr(corrects,guesses)[0]


if __name__ == '__main__':

    import dependency_tree as tr     
    trainTrees = tr.loadTrees("train")
    print "train number %d"%len(trainTrees)
     
    numW = len(tr.loadWordMap())

    relMap = tr.loadRelMap()
    relNum = len(relMap)
    word2vecs = tr.loadWord2VecMap()

    wvecDim = 100
    outputDim = 5

    rnn = depTreeRnnModel(relNum, wvecDim, outputDim)
    rnn.initialParams(word2vecs)
    minibatch = 200

    devTrees = tr.loadTrees("dev")

    best_dev_score  = 0.

    print "training model"
    for e in range(1000):
        print "iter ", e
        #print "Running epoch %d"%e
        classifier = rnn.train(trainTrees, minibatch)
        #print "Time per epoch : %f"%(end-start)
        
        dev_score = rnn.predict(devTrees,classifier)
        print "dev score is: %f"%dev_score

        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print "best dev score is: %f"%best_dev_score

    

    

