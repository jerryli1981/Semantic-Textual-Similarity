import numpy as np
import time
import sys,os,cPickle
from multiprocessing import Pool

from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
import nltk.classify.util
from sklearn import linear_model

from scipy.stats import pearsonr

""" softmax function for multi class logistic regression """
def softmax(W,x):
       vec=np.dot(W.T,x)
       vec1=np.exp(vec)
       res=vec1.T/np.sum(vec1,axis=0)
       return res.T;

def sigmoid(x):

    z = 1.0 / (1.0 + np.exp((-1) * x))
    return z

# derivative of normalized tanh
def dtanh(x):
    norm = np.linalg.norm(x)
    y = x - np.power(x, 3)
    dia = np.diag((1 - np.square(x)).flatten()) / norm
    pro = y.dot(x.T) / np.power(norm, 3)
    out = dia - pro
    return out


def forward_prop(params, tree, d, n_labels, training=True):

    tree.reset_finished()

    to_do = tree.get_nodes()

    (Wr_dict, Wv, b, We, Ws) = params
    
    # forward prop
    while to_do:
        
        curr = to_do.pop(0)

        # node is leaf
        if len(curr.kids) == 0:
            
            # activation function is the normalized tanh
            curr.p = np.tanh(Wv.dot(curr.vec) + b)
            curr.p_norm = curr.p / np.linalg.norm(curr.p)
            curr.ans_error = 0.0 #input no error

        else:

            # - root isn't a part of this! 
            # - more specifically, the stanford dep. parser creates a superficial ROOT node
            #   associated with the word "root" that we don't want to consider during training
            if len(to_do) == 0:
                
                ind, rel = curr.kids[0]
                curr.p = tree.get(ind).p
                curr.p_norm = tree.get(ind).p_norm
                curr.ans_error = 0.
                continue
            
            # check if all kids are finished
            all_done = True
            for ind, rel in curr.kids:
                if tree.get(ind).finished == 0:
                    all_done = False
                    break

            # if not, push the node back onto the queue
            if not all_done:
                to_do.append(curr)
                continue

            # otherwise, compute p at node
            else:
                kid_sum = np.zeros( (d, 1) )
                for ind, rel in curr.kids:
                    curr_kid = tree.get(ind)

                    try:
                        kid_sum += Wr_dict[rel].dot(curr_kid.p_norm)

                    # - this shouldn't happen unless the parser spit out a seriously 
                    #   malformed tree
                    except KeyError:
                        pass
                        #print 'forward propagation error'
                        #print tree.get_words()
                        #print curr.word, rel, tree.get(ind).word
                
                kid_sum += Wv.dot(curr.vec)
                curr.p = np.tanh(kid_sum + b)
                curr.p_norm = curr.p / np.linalg.norm(curr.p)
            
        if training:
            #cross entropy error
            if n_labels == 3:
                h = softmax(Ws, curr.p_norm)
    
                y = tree.label
                if y == 0:
                    Y = np.array([1,0,0], int).reshape(3,1)
                elif y == 1:
                    Y = np.array([0,1,0], int).reshape(3,1)
                elif y ==2:
                    Y = np.array([0,0,1], int).reshape(3,1)
                
                curr.ans_delta = Ws.dot(h-Y)
                
                h = h.reshape(3)
                h = np.log(h)
                cee = 0.
                for y in range(3):
                    if y == 0:
                        Y = np.array([1,0,0], int).reshape(3)
                    elif y == 1:
                        Y = np.array([0,1,0], int).reshape(3)
                    elif y ==2:
                        Y = np.array([0,0,1], int).reshape(3)
                        
                    cee +=h.dot(Y) 
                        
                curr.ans_error = -cee
                
            elif n_labels == 2:
                h = softmax(Ws, curr.p_norm)
    
                y = tree.label
                if y == 0:
                    Y = np.array([1,0], int).reshape(2,1)
                elif y == 1:
                    Y = np.array([0,1], int).reshape(2,1)
                
                curr.ans_delta = Ws.dot(h-Y)
                
                h = h.reshape(2)
                h = np.log(h)
                cee = 0.
                for y in range(2):
                    if y == 0:
                        Y = np.array([1,0], int).reshape(2)
                    elif y == 1:
                        Y = np.array([0,1], int).reshape(2)
                        
                    cee +=h.dot(Y) 
                        
                curr.ans_error = -cee
                
        curr.finished = 1
    
# computes gradients for the given tree and increments existing gradients
def backprop(params, tree, d, grads):

    (Wr_dict, Wv, b, We, Ws) = params

    # start with root's immediate kid (for same reason as forward prop)
    ind, rel = tree.get(0).kids[0]
    root = tree.get(ind)

    # operate on tuples of the form (node, parent delta)
    to_do = [ (root, np.zeros( (d, 1) ) ) ]


    while to_do:
        
        curr = to_do.pop()
        node = curr[0]
        pd = curr[1] #pd: parent delta

        # internal node
        if len(node.kids) > 0:

            act = pd+ node.ans_delta
            df = dtanh(node.p)
            node.delta_i = df.dot(act)

            for ind, rel in node.kids:

                curr_kid = tree.get(ind)
                grads[0][rel] += node.delta_i.dot(curr_kid.p_norm.T)
                to_do.append( (curr_kid, Wr_dict[rel].T.dot(node.delta_i) ) )

            grads[1] += node.delta_i.dot(node.vec.T)
            grads[2] += node.delta_i
            grads[3][node.ind, :] += Wv.T.dot(node.delta_i).ravel()
            grads[4] += node.delta_i
            
        # leaf
        else:
            act = pd + node.ans_delta
            df = dtanh(node.p)

            node.delta_i = df.dot(act)
            grads[1] += node.delta_i.dot(node.vec.T)
            grads[2] += node.delta_i
            grads[3][node.ind, :] += Wv.T.dot(node.delta_i).ravel()
            grads[4] += node.delta_i
            
# splits the training data into minibatches
# multi-core parallelization
def par_objective(num_proc, data, params, d, We_size, rel_dict, lambdas, n_labels):
    pool = Pool(processes=num_proc) 

    # non-data params
    oparams = [params, d, We_size, rel_dict, n_labels]
    
    # chunk size
    n = len(data) / num_proc
    split_data = [data[i:i+n] for i in range(0, len(data), n)]
    to_map = []
    for item in split_data:
        to_map.append( (oparams, item) )
        
    result = pool.map(objective_and_grad, to_map)
    pool.close()   # no more processes accepted by this pool    
    pool.join()    # wait until all processes are finished
    
    total_err = 0.0
    all_nodes = 0.0
    total_grad = None
    
    for (err, grad, num_nodes) in result:
        total_err += err

        if total_grad is None:
            total_grad = grad
        else:
            total_grad += grad

        all_nodes += num_nodes


    # add L2 regularization
    params = unroll_params(params, d, We_size, rel_dict, n_labels)
    (Wr_dict, Wv, b, L, Ws) = params  
    grads = unroll_params(total_grad, d, We_size, rel_dict, n_labels)
    [lambda_W, lambda_L] = lambdas

    reg_cost = 0.0
    for key in rel_dict:
        reg_cost += 0.5 * lambda_W * np.sum(Wr_dict[key] ** 2)
        grads[0][key] = grads[0][key] / all_nodes
        grads[0][key] += lambda_W * Wr_dict[key]

    reg_cost += 0.5 * lambda_W * np.sum(Wv ** 2)
    grads[1] = grads[1] / all_nodes
    grads[1] += lambda_W * Wv

    grads[2] = grads[2] / all_nodes

    reg_cost += 0.5 * lambda_L * np.sum(L ** 2)
    grads[3] = grads[3] / all_nodes
    grads[3] += lambda_L * L
    
    grads[4] = grads[4] / all_nodes

    cost = total_err / all_nodes + reg_cost
        
    grad = roll_params(grads, rel_dict)

    return cost, grad


def roll_params(params, rel_dict):
      
    (Wr_dict, Wv, b, We, Ws) = params

    rels = np.concatenate( [Wr_dict[key].ravel() for key in rel_dict] )
    return np.concatenate( (rels, Wv.ravel(), b.ravel(), We.ravel(), Ws.ravel() ) )

def unroll_params(arr, d, We_size, rel_dict, n_labels):

    mat_size = d * d
    Wr_dict = {}
    ind = 0

    for r in rel_dict:
        Wr_dict[r] = arr[ind: ind + mat_size].reshape( (d, d) )
        ind += mat_size

    Wv = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size

    b = arr[ind : ind + d].reshape( (d, 1) )
    ind += d

    We = arr[ind : ind + We_size * d].reshape( (We_size, d))
    ind += We_size *d
    
    Ws = arr[ind : ind + d * n_labels].reshape( (d, n_labels) )

    return [Wr_dict, Wv, b, We, Ws]


# this function computes the objective / grad for each minibatch
def objective_and_grad(par_data):

    params, d, We_size, rel_dict, n_labels = par_data[0]
    data = par_data[1]
    params = unroll_params(params, d, We_size, rel_dict, n_labels)
    (Wr_dict, Wv, b, We, Ws) = params
    
    rel_grads = {}
    for rel in rel_dict:
        rel_grads[rel] = np.zeros( (d, d) )

    grads =  [
        rel_grads,
        np.zeros((d, d)), #Wv
        np.zeros((d, 1)), #Bs
        np.zeros((We_size, d)), #We
        np.zeros((d, n_labels)) #Ws
        ]
    
    error_sum = 0.0
    num_nodes = 0
    tree_size = 0

    # compute error and gradient for each tree in minibatch
    # also keep track of total number of nodes in minibatch
    for index, tree in enumerate(data):

        nodes = tree.get_nodes()
        for node in nodes:
            node.vec = We[node.ind, :].reshape((d, 1))
        forward_prop(params, tree, d, n_labels)
        error_sum += tree.error()
        tree_size += len(nodes)
        backprop(params, tree, d, grads)
        
    #print error_sum

    (rel_grads, Wv_grad, b_grad, We_grad, Ws_grad) = grads
    
    grad = roll_params(grads, rel_dict)
   
    return (error_sum, grad, tree_size)

class Adagrad(): 

    def __init__(self, dim):
        self.dim = dim
        self.eps = 1e-3

        # initial learning rate
        self.learning_rate = 0.05

        # stores sum of squared gradients 
        self.h = np.zeros(self.dim)

    def rescale_update(self, gradient):
        curr_rate = np.zeros(self.h.shape)
        self.h += gradient ** 2
        curr_rate = self.learning_rate / (np.sqrt(self.h) + self.eps)
        return curr_rate * gradient

    def reset_weights(self):
        self.h = np.zeros(self.dim)
        
def validate_linear(trainTrees, validTrees, params, d):
    
    stop = stopwords.words('english')

    (Wr_dict, Wv, bias, We, Ws) = params
    
    for tree in trainTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
    
    for tree in validTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
            
    train_feats = []  
    train_feats_x = []
    train_feats_y = []      
    for num_finished, tree in enumerate(trainTrees):
    
            # process validation trees
            forward_prop(params, tree, d, training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            curr_feats_x = []
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                curr_feats_x.append(val)
                
            curr_feats_x = np.asarray(curr_feats_x)
            train_feats.append( (curr_feats, tree.score) )
            train_feats_x.append(curr_feats_x)
            train_feats_y.append(float(tree.score))
            
    train_feats_x = np.asarray(train_feats_x)        
    train_feats_y = np.asarray(train_feats_y)        
            
    val_feats = []   
    val_feats_x = []
    val_feats_y = []     
    for num_finished, tree in enumerate(validTrees):
    
            # process validation trees
            forward_prop(params, tree, d, training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            curr_feats_x = []
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                curr_feats_x.append(val)
                
            curr_feats_x = np.asarray(curr_feats_x)    
            val_feats.append( (curr_feats, tree.score) )
            val_feats_x.append(curr_feats_x)
            val_feats_y.append(float(tree.score))
            
    val_feats_x = np.asarray(val_feats_x)        
    val_feats_y = np.asarray(val_feats_y)
            

    print 'training'
    regr = linear_model.LinearRegression()
    regr.fit(train_feats_x, train_feats_y)
    
    #classifier = SklearnClassifier(LogisticRegression(C=10))
    #classifier.train(train_feats)   
    
    print 'predicting...'
    #train_acc = nltk.classify.util.accuracy(classifier, train_feats)
    #val_acc = nltk.classify.util.accuracy(classifier, val_feats)
    #return train_acc, 1-val_acc
    #return 0, np.mean((regr.predict(val_feats_x)-val_feats_y) ** 2)

    return 0, 1-pearsonr(regr.predict(val_feats_x),val_feats_y)[0]

        
# train a logistic regression classifier       
def validate(trainTrees, validTrees, params, d, n_labels):
    
    stop = stopwords.words('english')

    (Wr_dict, Wv, bias, We, Ws) = params
    
    for tree in trainTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
    
    for tree in validTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
            
    train_feats = []        
    for num_finished, tree in enumerate(trainTrees):
    
            # process validation trees
            forward_prop(params, tree, d, n_labels, training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                
            train_feats.append( (curr_feats, tree.label) )
            
    val_feats = []        
    for num_finished, tree in enumerate(validTrees):
    
            # process validation trees
            forward_prop(params, tree, d, n_labels, training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                
            val_feats.append( (curr_feats, tree.label) )
            

    print 'training'
    classifier = SklearnClassifier(LogisticRegression(C=10))
    classifier.train(train_feats)   
    
    print 'predicting...'
    train_acc = nltk.classify.util.accuracy(classifier, train_feats)
    val_acc = nltk.classify.util.accuracy(classifier, val_feats)
    return train_acc, 1-val_acc

def test_linear(trainTrees, testTrees, params, d):
    
    stop = stopwords.words('english')

    (Wr_dict, Wv, bias, We, Ws) = params
    
    for tree in trainTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
            
    
    for tree in testTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
            
    train_feats = []  
    train_feats_x = []
    train_feats_y = []      
    for num_finished, tree in enumerate(trainTrees):
    
            # process validation trees
            forward_prop(params, tree, d, training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            curr_feats_x = []
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                curr_feats_x.append(val)
                
            curr_feats_x = np.asarray(curr_feats_x)
            train_feats.append( (curr_feats, tree.score) )
            train_feats_x.append(curr_feats_x)
            train_feats_y.append(float(tree.score))
            
    train_feats_x = np.asarray(train_feats_x)        
    train_feats_y = np.asarray(train_feats_y)    
                       
    test_feats = []   
    test_feats_x = []
    test_feats_y = []     
    for num_finished, tree in enumerate(testTrees):
    
            # process validation trees
            forward_prop(params, tree, d, training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            curr_feats_x = []
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                curr_feats_x.append(val)
                
            curr_feats_x = np.asarray(curr_feats_x)    
            test_feats.append( (curr_feats, tree.score) )
            test_feats_x.append(curr_feats_x)
            test_feats_y.append(float(tree.score))
            
    test_feats_x = np.asarray(test_feats_x)        
    test_feats_y = np.asarray(test_feats_y)
            

    print 'training'
    regr = linear_model.LinearRegression()
    regr.fit(train_feats_x, train_feats_y)
    
    #classifier = SklearnClassifier(LogisticRegression(C=10))
    #classifier.train(train_feats)   
    
    print 'predicting...'
    #train_acc = nltk.classify.util.accuracy(classifier, train_feats)
    #val_acc = nltk.classify.util.accuracy(classifier, val_feats)
    #return train_acc, 1-val_acc
    return 1-pearsonr(regr.predict(test_feats_x),test_feats_y)[0]

def test(trainTrees, testTrees, params, d, n_labels):
    
    stop = stopwords.words('english')

    (Wr_dict, Wv, bias, We, Ws) = params
    
    for tree in trainTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
            
    
    for tree in testTrees:
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((d, 1))
            
    train_feats = []        
    for num_finished, tree in enumerate(trainTrees):
    
            # process validation trees
            forward_prop(params, tree, d, n_labels, training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                
            train_feats.append( (curr_feats, tree.label) )
                       
    test_feats = []        
    for num_finished, tree in enumerate(testTrees):
    
            # process validation trees
            forward_prop(params, tree, d, n_labels,training=False)
    
            ave = np.zeros( (d, 1))
            words = np.zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):
    
                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1
    
            ave = ave / count
            featvec = ave.flatten()
    
            curr_feats = {}
            for dim, val in np.ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val
                
            test_feats.append( (curr_feats, tree.label) )
            

    print 'training'
    classifier = SklearnClassifier(LogisticRegression(C=10))
    classifier.train(train_feats)   
    
    print 'predicting...'
    test_acc = nltk.classify.util.accuracy(classifier, test_feats)
    return 1-test_acc
         
def evaluate_DT_RNN(revs, partition, W, rel_dict, batch_size, n_labels, n_epochs=200, d = 300):
    
    trainTrees = []
    testTrees = []
    validTrees = []
    for i in range(len(revs)):    
        rev = revs[i]
        tree = rev["tree"]

        if partition[i] == "Train":
            trainTrees.append(tree)
        elif partition[i] == "Valid":
            validTrees.append(tree)
        elif partition[i] == "Test":
            testTrees.append(tree)
                     
    ## remove incorrectly parsed sentences from data
    # print 'removing bad trees train...'
    bad_trees = []
    for ind, tree in enumerate(trainTrees):
        if len(tree.nodes) ==0 or tree.get(0).is_word == 0:
            print tree.get_words(), ind
            bad_trees.append(ind)

    # pop bad trees, higher indices first
    # print 'removed ', len(bad_trees)
    for ind in bad_trees[::-1]:
        trainTrees.pop(ind)

    # print 'removing bad trees val...'
    bad_trees = []
    for ind, tree in enumerate(validTrees):

        if len(tree.nodes) ==0 or tree.get(0).is_word == 0:
            # print tree.get_words(), ind
            bad_trees.append(ind)

    # pop bad trees, higher indices first
    # print 'removed ', len(bad_trees)
    for ind in bad_trees[::-1]:
        validTrees.pop(ind)
    
    
    r = np.sqrt(6) / np.sqrt(201)
    Wr_dict = {}
    for rel in rel_dict:
        Wr_dict[rel] = np.random.rand(d, d) * 2 * r - r
        
    Wv = np.random.rand(d, d) * 2 * r - r  
    bias = np.zeros((d, 1))
    
    We = W[:, :d]
   
    Ws = np.random.rand(d, n_labels) * 2 * r - r
    
    params = (Wr_dict, Wv, bias, We, Ws)
    rps = roll_params(params, rel_dict)

    num_proc = 4
    
    lambdas = [1e-4, 1e-3]
    
    ag = Adagrad(rps.shape)
    
    print '... training'    
    best_validation_loss = np.inf
    best_test_lost = np.inf
    start_time = time.clock()
    epoch = 0
    
    min_error = float('inf')

    while epoch < n_epochs:
        epoch = epoch + 1
        print 'epoch @ epoch = ', epoch
        
        np.random.shuffle(trainTrees)
        batches = [trainTrees[x : x + batch_size] for x in xrange(0, len(trainTrees), batch_size)]

        epoch_error = 0.0
        for batch_ind, batch in enumerate(batches):

            cost, grad = par_objective(num_proc, batch, rps, d, We.shape[0], rel_dict, lambdas, n_labels)
            update = ag.rescale_update(grad)
            rps = rps - update
            epoch_error += cost
            #print cost
                 
        """
        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            savedparams = unroll_params(rps, d, len(vocab), rel_dict)
        """  
            
        # reset adagrad weights
        if epoch % 3 == 0 and epoch != 0:
            ag.reset_weights()
            
        
        params = unroll_params(rps, d, We.shape[0], rel_dict, n_labels)
        
        train_acc, this_validation_loss = validate(trainTrees, validTrees, params, d, n_labels)
        #train_acc, this_validation_loss = validate_linear(trainTrees, validTrees, params, d)
   
        print('epoch %i, validation error %f %%' %
                      (epoch, this_validation_loss * 100.))
             
        # if we got the best validation score until now
        if this_validation_loss <= best_validation_loss:
        #if True:

            best_validation_loss = this_validation_loss           
            print('epoch %i, best validation loss %f %%' %
                      (epoch, best_validation_loss * 100.)) 
           
            savedparams = unroll_params(rps, d, We.shape[0], rel_dict, n_labels)
            test_loss = test(trainTrees, testTrees, params, d, n_labels)
            #test_loss = test_linear(trainTrees, testTrees, params, d)
            
            print('epoch %i, test error of best model %f %%' % (epoch,  test_loss * 100.) )
            
            if test_loss < best_test_lost:
                best_test_lost = test_loss
            

                                                                                                
    end_time = time.clock()

    print('Best validation loss of %f %% , '
          'with test loss %f %%' %
          (best_validation_loss * 100., best_test_lost * 100.))
    
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    return 1-best_validation_loss,  1-best_test_lost, savedparams
    
