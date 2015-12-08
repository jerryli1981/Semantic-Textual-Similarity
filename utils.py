import numpy as np
import os
import theano.tensor as T

import networkx as nx

import theano


# Caution: Setting this to true prevents a model to be stored correctly. Loading
# is not possible. This is because the Print function is not properly
# serialized.
PRINT_VARS = False

def debug_print(var, name):
    """Wrap the given Theano variable into a Print node for debugging.
    If the variable is wrapped into a Print node depends on the state of the
    PRINT_VARS variable above. If it is false, this method just returns the
    original Theano variable.
    The given variable is printed to console whenever it is used in the graph.
    Parameters
    ----------
    var : Theano variable
        variable to be wrapped
    name : str
        name of the variable in the console output
    Returns
    -------
    Theano variable
        wrapped Theano variable
    Example
    -------
    import theano.tensor as T
    d = T.dot(W, x) + b
    d = debug_print(d, 'dot_product')
    """

    if PRINT_VARS is False:
        return var

    return theano.printing.Print(name)(var)

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

def logsoftmax(x):
    N = x.shape[0]
    x -= np.max(x,axis=1).reshape(N,1)
    x = np.exp(x)/np.sum(np.exp(x),axis=1).reshape(N,1)
    return np.log(x)

def sigmoid(x):
    """ Sigmoid function """
    x = 1/(1+np.exp(-x))    
    return x

def derivative_sigmoid(f):
    """ Sigmoid gradient function """
    f = f*(1-f)
    return f

def derivative_softmax(f):
    """ Softmax gradient function """
    f = f*(1-f)
    return f

# derivative of normalized tanh
def derivative_norm_tanh(x):
    norm = np.linalg.norm(x)
    y = x - np.power(x, 3)
    dia = np.diag((1 - np.square(x)).flatten()) / norm
    pro = y.dot(x.T) / np.power(norm, 3)
    out = dia - pro
    return out

def norm_tanh(x):
    x = np.tanh(x)
    x /= np.linalg.norm(x)

    return x

def derivative_tanh(f):
	return 1 - f**2

def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi.reshape((1,1))    

def pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x = x - np.mean(x)
    y = y- np.mean(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def iterate_minibatches(inputs1, inputs1_mask, inputs2, inputs2_mask, 
    targets, scores, scores_pred, batchsize, shuffle=False):
    assert len(inputs1) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs1[excerpt], inputs1_mask[excerpt], inputs2[excerpt], inputs2_mask[excerpt], targets[excerpt], scores[excerpt], scores_pred[excerpt]

def iterate_minibatches_(inputs, batchsize, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield ( input[excerpt] for input in inputs )

def iterate_minibatches_tree(dataset, batchsize, shuffle=False):

    l_trees, r_trees, Y_labels, Y_scores, Y_scores_pred = dataset

    if shuffle:
        indices = np.arange(len(l_trees))
        np.random.shuffle(indices)
    else:
        indices = np.arange(len(l_trees))

    batches = []
    for start_idx in range(0, len(dataset[0]) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        l_t_batch=[]
        r_t_batch=[]
        y_l_batch=[]
        y_s_batch = np.zeros((len(excerpt)), dtype=np.float32)
        y_s_pred_batch=[]
        for id, i in enumerate(excerpt):
            l_t_batch.append(l_trees[i])
            r_t_batch.append(r_trees[i])
            y_l_batch.append(Y_labels[i])
            y_s_batch[id] = Y_scores[i]
            y_s_pred_batch.append(Y_scores_pred[i])

        batches.append((l_t_batch, r_t_batch, y_l_batch, y_s_batch, y_s_pred_batch))

    return batches


def loadWord2VecMap(word2vec_path):
    import cPickle as pickle
    
    with open(word2vec_path,'r') as fid:
        return pickle.load(fid)

#this method special for main_keras_graph builde with embedding
def read_sequence_dataset_embedding(dataset_dir, dataset_name, wordEmbeddings, maxlen=36):

    labelIdx_m = {"NEUTRAL":2, "ENTAILMENT":1, "CONTRADICTION":0}

    a_s = os.path.join(dataset_dir, dataset_name+"/a.toks")
    b_s = os.path.join(dataset_dir, dataset_name+"/b.toks")
    sims = os.path.join(dataset_dir, dataset_name+"/sim.txt")
    labs = os.path.join(dataset_dir, dataset_name+"/label.txt") 

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    Y_scores_pred = np.zeros((data_size, 6), dtype=np.float32)    
    Y_scores = np.zeros((data_size), dtype=np.float32) 
    labels = []

    X1 = np.zeros((data_size, maxlen, 300), dtype=np.int16)
    X2 = np.zeros((data_size, maxlen, 300), dtype=np.int16)

    X1_mask = np.zeros((data_size, maxlen), dtype=np.int16)
    X2_mask = np.zeros((data_size, maxlen), dtype=np.int16)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["<UNK>"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        vocab[word] = idx

    with open(a_s, "rb") as f1, \
         open(b_s, "rb") as f2, \
         open(sims, "rb") as f3, \
         open(labs, 'rb') as f4:
                        
        for i, (a, b, sim, ent) in enumerate(zip(f1,f2,f3,f4)):

            a = a.rstrip('\n')
            b = b.rstrip('\n')
            sim = float(sim.rstrip('\n'))
            ent = ent.rstrip('\n')

            ceil = np.ceil(sim)
            floor = np.floor(sim)
            if ceil == floor:
                Y_scores_pred[i, floor] = 1
            else:
                Y_scores_pred[i, floor] = ceil-sim
                Y_scores_pred[i, ceil] = sim-floor

            label = labelIdx_m[ent]
            Y_scores[i] = 0.25 * (sim -1)
            labels.append(label)

            toks_a = a.split()
            toks_b = b.split()

            for j in range(maxlen):
                if j < maxlen - len(toks_a):
                    X1[i,j] = wordEmbeddings[:,vocab["<UNK>"]]
                    X1_mask[i, j] = 0
                else:
                    idx = vocab[toks_a[j-maxlen+len(toks_a)]]
                    X1[i, j] = wordEmbeddings[:,idx]
                    X1_mask[i, j] = 1

                    
            for j in range(maxlen):
                if j < maxlen - len(toks_b):
                    X2[i,j] = wordEmbeddings[:,vocab["<UNK>"]]
                    X2_mask[i, j] = 0
                else:
                    idx = vocab[toks_b[j-maxlen+len(toks_b)]]
                    X2[i,j] = wordEmbeddings[:,idx]
                    X2_mask[i, j] = 1
      
    Y_scores_pred = Y_scores_pred[:, 1:]
    Y_labels = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return X1, X1_mask, X2, X2_mask, Y_labels, Y_scores, Y_scores_pred

def read_sequence_dataset(dataset_dir, dataset_name, maxlen=36):

    labelIdx_m = {"NEUTRAL":2, "ENTAILMENT":1, "CONTRADICTION":0}

    a_s = os.path.join(dataset_dir, dataset_name+"/a.toks")
    b_s = os.path.join(dataset_dir, dataset_name+"/b.toks")
    sims = os.path.join(dataset_dir, dataset_name+"/sim.txt")
    labs = os.path.join(dataset_dir, dataset_name+"/label.txt") 

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    Y_scores_pred = np.zeros((data_size, 6), dtype=np.float32)    
    Y_scores = np.zeros((data_size), dtype=np.float32) 
    labels = []

    X1 = np.zeros((data_size, maxlen), dtype=np.int16)
    X2 = np.zeros((data_size, maxlen), dtype=np.int16)

    X1_mask = np.zeros((data_size, maxlen), dtype=np.int16)
    X2_mask = np.zeros((data_size, maxlen), dtype=np.int16)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["<UNK>"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        vocab[word] = idx

    with open(a_s, "rb") as f1, \
         open(b_s, "rb") as f2, \
         open(sims, "rb") as f3, \
         open(labs, 'rb') as f4:
                        
        for i, (a, b, sim, ent) in enumerate(zip(f1,f2,f3,f4)):

            a = a.rstrip('\n')
            b = b.rstrip('\n')
            sim = float(sim.rstrip('\n'))
            ent = ent.rstrip('\n')

            ceil = np.ceil(sim)
            floor = np.floor(sim)
            if ceil == floor:
                Y_scores_pred[i, floor] = 1
            else:
                Y_scores_pred[i, floor] = ceil-sim
                Y_scores_pred[i, ceil] = sim-floor

            label = labelIdx_m[ent]
            Y_scores[i] = 0.25 * (sim -1)
            labels.append(label)

            toks_a = a.split()
            toks_b = b.split()

            for j in range(maxlen):
                if j < maxlen - len(toks_a):
                    X1[i,j] = vocab["<UNK>"]
                    X1_mask[i, j] = 0
                else:
                    X1[i, j] = vocab[toks_a[j-maxlen+len(toks_a)]]
                    X1_mask[i, j] = 1

                    
            for j in range(maxlen):
                if j < maxlen - len(toks_b):
                    X2[i,j] = vocab["<UNK>"]
                    X2_mask[i, j] = 0
                else:
                    X2[i,j] = vocab[toks_b[j-maxlen+len(toks_b)]]
                    X2_mask[i, j] = 1
      
    Y_scores_pred = Y_scores_pred[:, 1:]
    Y_labels = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return X1, X1_mask, X2, X2_mask, Y_labels, Y_scores, Y_scores_pred

def read_tree_dataset(data_dir, name, rangeScores=5, numLabels=3, maxlen=36):

    labelIdx_m = {"NEUTRAL":2, "ENTAILMENT":1, "CONTRADICTION":0}

    a_s = os.path.join(data_dir, name+"/a.toks")
    b_s = os.path.join(data_dir, name+"/b.toks")
    sims = os.path.join(data_dir, name+"/sim.txt")
    labs = os.path.join(data_dir, name+"/label.txt") 
    a_p = os.path.join(data_dir, name+"/a.parents")
    b_p = os.path.join(data_dir, name+"/b.parents")
    a_r = os.path.join(data_dir, name+"/a.rels")
    b_r = os.path.join(data_dir, name+"/b.rels")

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    Y_scores_pred = np.zeros((data_size, rangeScores+1), dtype=np.float32)    
    Y_scores = np.zeros((data_size), dtype=np.float32) 
    labels = []

    l_trees = []
    r_trees = []

    from collections import defaultdict
    words = defaultdict(int)

    from dependency_tree import DTree

    vocab_path = os.path.join(data_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["<UNK>"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        vocab[word] = idx

    rel_vocab_path = os.path.join(data_dir, 'rel_vocab.txt')

    rels = defaultdict(int)

    with open(rel_vocab_path, 'r') as f:
        for tok in f:
            rels[tok.rstrip('\n')] += 1

    rel_vocab = dict(zip(rels.iterkeys(),xrange(len(rels))))

    with open(a_s, 'rb') as f1, \
         open(b_s, 'rb') as f2, \
         open(sims, 'rb') as f3, \
         open(labs, 'rb') as f4, \
         open(a_p, 'rb') as f5, \
         open(b_p, 'rb') as f6, \
         open(a_r, 'rb') as f7, \
         open(b_r, 'rb') as f8:                   

        for i, (a, b, sim, ent, a_p, b_p, a_r, b_r) in enumerate(zip(f1,f2,f3,f4,f5,f6,f7,f8)):

            a = a.rstrip('\n')
            b = b.rstrip('\n')
            sim = float(sim.rstrip('\n'))
            ent = ent.rstrip('\n')

            ceil = np.ceil(sim)
            floor = np.floor(sim)
            if ceil == floor:
                Y_scores_pred[i, floor] = 1
            else:
                Y_scores_pred[i, floor] = ceil-sim 
                Y_scores_pred[i, ceil] = sim-floor

            label = labelIdx_m[ent]
            Y_scores[i] = 0.25 * (sim -1)

            labels.append(label)

            toks_a = a.split()
            toks_b = b.split()

            a_p_l = a_p.rstrip('\n').split()
            b_p_l = b_p.rstrip('\n').split()
            a_r_l = a_r.rstrip('\n').split()
            b_r_l = b_r.rstrip('\n').split()

            dep_tree_a = DTree(toks_a, a_p_l, a_r_l,vocab,rel_vocab)
            dep_tree_b = DTree(toks_b, b_p_l, b_r_l,vocab,rel_vocab)

            l_trees.append(dep_tree_a)
            r_trees.append(dep_tree_b)


    Y_scores_pred = Y_scores_pred[:, 1:]
    Y_labels = np.zeros((len(labels), numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return l_trees, r_trees, Y_labels, Y_scores, Y_scores_pred

def merge_tree_dataset(data_dir, name):

    a_s = os.path.join(data_dir, name+"/a.toks")
    b_s = os.path.join(data_dir, name+"/b.toks")

    a_p = os.path.join(data_dir, name+"/a.parents")
    b_p = os.path.join(data_dir, name+"/b.parents")
    a_r = os.path.join(data_dir, name+"/a.rels")
    b_r = os.path.join(data_dir, name+"/b.rels")

    m_p = os.path.join(data_dir, name+"/m.parents")

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    l_trees = []
    r_trees = []

    from collections import defaultdict
    words = defaultdict(int)

    from dependency_tree import DTree

    vocab_path = os.path.join(data_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["<UNK>"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        vocab[word] = idx

    rel_vocab_path = os.path.join(data_dir, 'rel_vocab.txt')

    rels = defaultdict(int)

    with open(rel_vocab_path, 'r') as f:
        for tok in f:
            rels[tok.rstrip('\n')] += 1

    rel_vocab = dict(zip(rels.iterkeys(),xrange(len(rels))))



    with open(a_s, 'rb') as f1, \
         open(b_s, 'rb') as f2, \
         open(a_p, 'rb') as f5, \
         open(b_p, 'rb') as f6, \
         open(a_r, 'rb') as f7, \
         open(b_r, 'rb') as f8,\
         open(m_p, 'wb') as f9:              

        for i, (a, b, a_p, b_p, a_r, b_r) in enumerate(zip(f1,f2,f5,f6,f7,f8)):

            a = a.rstrip('\n')
            b = b.rstrip('\n')

            toks_a = a.split()
            toks_b = b.split()

            a_p_l = a_p.rstrip('\n').split()
            b_p_l = b_p.rstrip('\n').split()
            a_r_l = a_r.rstrip('\n').split()
            b_r_l = b_r.rstrip('\n').split()

            dep_tree_a = DTree(toks_a, a_p_l, a_r_l,vocab,rel_vocab)
            dep_tree_b = DTree(toks_b, b_p_l, b_r_l,vocab,rel_vocab)

            dep_tree_a.mergeWith(dep_tree_b)

            G = nx.DiGraph()
            G.add_edges_from(dep_tree_a.dependencies)

            is_dag = nx.is_directed_acyclic_graph(G)

            if is_dag:

                node_size = len(dep_tree_a.nodes)
                parents = [-1] * node_size
                for govIdx, depIdx in dep_tree_a.dependencies:
                    parents[depIdx - 1] = str(govIdx)

                assert node_size == len(dep_tree_a.dependencies)
                f9.write(' '.join(parents) + '\n')
            else:
                print 'is not dag'

if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    merge_tree_dataset(sick_dir, "train")
    merge_tree_dataset(sick_dir, "dev")
    merge_tree_dataset(sick_dir, "test")
    print 'done'


