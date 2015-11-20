import numpy as np
import os

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

def pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x = x - np.mean(x)
    y = y- np.mean(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def iterate_minibatches(inputs1, inputs2, targets, scores, scores_pred, batchsize, shuffle=False):
    assert len(inputs1) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs1[excerpt], inputs2[excerpt], targets[excerpt], scores[excerpt], scores_pred[excerpt]

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


def load_data_embedding(data, wordEmbeddings, args, maxlen=36):

    Y_scores_pred = np.zeros((len(data), args.rangeScores+1), dtype=np.float32)

    #maxlen = 0
    for i, (label, score, l_t, r_t) in enumerate(data):
        """
        max_ = max(len(l_t.nodes), len(r_t.nodes))
        if maxlen < max_:
            maxlen = max_
        """
        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y_scores_pred[i, floor] = 1
        else:
            Y_scores_pred[i, floor] = ceil-sim 
            Y_scores_pred[i, ceil] = sim-floor

    Y_scores_pred = Y_scores_pred[:, 1:]

    X1 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)
    X2 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)

    Y_labels = np.zeros((len(data)), dtype=np.int32)
    Y_scores = np.zeros((len(data)), dtype=np.float32)

    #np.random.uniform(-0.25,0.25,k)

    for i, (label, score, l_tree, r_tree) in enumerate(data):

        for j, Node in enumerate(l_tree.nodes):
            X1[i, j] =  wordEmbeddings[:, Node.index]
            if j >= len(l_tree.nodes):
                X1[i,j] =  np.random.uniform(-0.25,0.25, args.wvecDim)

        for k, Node in enumerate(r_tree.nodes):
            X2[i, k] =  wordEmbeddings[:, Node.index]
            if j >= len(r_tree.nodes):
                X2[i, j] =  np.random.uniform(-0.25,0.25, args.wvecDim)

        Y_labels[i] = label
        Y_scores[i] = score
        
    return X1, X2, Y_labels, Y_scores, Y_scores_pred


def load_data_index(data, args, vocabSize, maxlen=36):

    Y_scores_pred = np.zeros((len(data), args.rangeScores+1), dtype=np.float32)

    #maxlen = 0
    for i, (label, score, l_t, r_t) in enumerate(data):
        """
        max_ = max(len(l_t.nodes), len(r_t.nodes))
        if maxlen < max_:
            maxlen = max_
        """
        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y_scores_pred[i, floor] = 1
        else:
            Y_scores_pred[i, floor] = ceil-sim 
            Y_scores_pred[i, ceil] = sim-floor

    Y_scores_pred = Y_scores_pred[:, 1:]

    Y_scores = np.zeros((len(data)), dtype=np.float32)

    labels = []
    X1 = np.zeros((len(data), maxlen), dtype=np.float32)
    X2 = np.zeros((len(data), maxlen), dtype=np.float32)

    for i, (label, score, l_t, r_t) in enumerate(data):

        for j in range(maxlen):
            if j < maxlen - len(l_t.nodes):
                X1[i,j] = vocabSize-1
            else:
                X1[i, j] =  l_t.nodes[j-maxlen+len(l_t.nodes)].index
                

        for j in range(maxlen):
            if j < maxlen - len(r_t.nodes):
                X2[i,j] = vocabSize-1
            else:
                X2[i, j] =  r_t.nodes[j-maxlen+len(r_t.nodes)].index

        labels.append(label)
        Y_scores[i] = score

    Y_labels = np.zeros((len(labels), args.numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return X1, X2, Y_labels, Y_scores, Y_scores_pred

def load_data_matrix(data, args, seq_len=36, n_children=6, unfinished_flag=-2):

    Y = np.zeros((len(data), args.outputDim+1), dtype=np.float32)
    scores = np.zeros((len(data)), dtype=np.float32)

    # to store hidden representation
    #(rootFlag, finishedFlag, globalgovIdx, n_children* (locaDepIdx, globalDepIdx, relIdx) , hiddenRep)
    storage_dim = 1 + 1 + 1 + 3*n_children + args.wvecDim

    X1 = np.zeros((len(data), seq_len, storage_dim), dtype=np.float32)
    X1.fill(-1.0)
    X2 = np.zeros((len(data), seq_len, storage_dim), dtype=np.float32)
    X2.fill(-1.0)
    
    for i, (score, item) in enumerate(data):
        first_t, second_t= item

        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y[i, floor] = 1
        else:
            Y[i, floor] = ceil-sim
            Y[i, ceil] = sim-floor

        f_idxSet = set()
        for govIdx, depIdx in first_t.dependencies:
            f_idxSet.add(govIdx)
            f_idxSet.add(depIdx)

        for j, Node in enumerate(first_t.nodes):

            if j not in f_idxSet:
                continue

            node_vec = np.zeros((storage_dim,), dtype=np.float32)
            node_vec.fill(-1.0)
            if j == first_t.rootIdx:
                node_vec[0] = 1

            node_vec[1] = unfinished_flag
            node_vec[2] = Node.index

            if len(Node.kids) != 0:

                r = range(0, 3*n_children, 3)
                r = r[:len(Node.kids)]
                for d, c in enumerate(r):
                    localDepIdx, rel = Node.kids[d]
                    node_vec[3+c] = localDepIdx
                    node_vec[4+c] = first_t.nodes[localDepIdx].index
                    node_vec[5+c] = rel.index


            X1[i, j] = node_vec


        s_idxSet = set()
        for govIdx, depIdx in second_t.dependencies:
            s_idxSet.add(govIdx)
            s_idxSet.add(depIdx)

        for j, Node in enumerate(second_t.nodes):

            if j not in s_idxSet:
                continue

            node_vec = np.zeros((storage_dim,), dtype=np.float32)
            node_vec.fill(-1.0)
            if j == second_t.rootIdx:
                node_vec[0] = 1

            node_vec[1] = unfinished_flag
            node_vec[2] = Node.index

            if len(Node.kids) != 0:

                r = range(0, 3*n_children, 3)
                r = r[:len(Node.kids)]
                for d, c in enumerate(r):
                    localDepIdx, rel = Node.kids[d]
                    node_vec[3+c] = localDepIdx
                    node_vec[4+c] = second_t.nodes[localDepIdx].index
                    node_vec[5+c] = rel.index

            X2[i, j] = node_vec
   
        scores[i] = score

    Y = Y[:, 1:]

    input_shape = (len(data), seq_len, storage_dim)
      
    return X1, X2, Y, scores, input_shape


def loadWord2VecMap(word2vec_path):
    import cPickle as pickle
    
    with open(word2vec_path,'r') as fid:
        return pickle.load(fid)


def read_dataset(data_dir, name, rangeScores=5, numLabels=3, maxlen=36):

    labelIdx_m = {"NEUTRAL":2, "ENTAILMENT":1, "CONTRADICTION":0}

    a_s = os.path.join(data_dir, name+"/a.toks")
    b_s = os.path.join(data_dir, name+"/b.toks")
    sims = os.path.join(data_dir, name+"/sim.txt")
    labs = os.path.join(data_dir, name+"/label.txt") 

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    Y_scores_pred = np.zeros((data_size, rangeScores+1), dtype=np.float32)    
    Y_scores = np.zeros((data_size), dtype=np.float32) 
    labels = []

    X1 = np.zeros((data_size, maxlen), dtype=np.float32)
    X2 = np.zeros((data_size, maxlen), dtype=np.float32)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(data_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = dict(zip(words.iterkeys(),xrange(len(words))))
    vocab["<UNK>"] = len(words) # Add unknown as word

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
                else:
                    X1[i, j] = vocab[toks_a[j-maxlen+len(toks_a)]]
                    
            for j in range(maxlen):
                if j < maxlen - len(toks_b):
                    X2[i,j] = vocab["<UNK>"]
                else:
                    X2[i,j] = vocab[toks_b[j-maxlen+len(toks_b)]]
      

    Y_scores_pred = Y_scores_pred[:, 1:]
    Y_labels = np.zeros((len(labels), numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return X1, X2, Y_labels, Y_scores, Y_scores_pred

def read_dataset_E(data_dir, name, rangeScores=5, numLabels=3, maxlen=36):

    wordEmbeddings = loadWord2VecMap(os.path.join(data_dir, 'word2vec.bin'))

    labelIdx_m = {"NEUTRAL":2, "ENTAILMENT":1, "CONTRADICTION":0}

    a_s = os.path.join(data_dir, name+"/a.toks")
    b_s = os.path.join(data_dir, name+"/b.toks")
    sims = os.path.join(data_dir, name+"/sim.txt")
    labs = os.path.join(data_dir, name+"/label.txt") 

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    Y_scores_pred = np.zeros((data_size, rangeScores+1), dtype=np.float32)    
    Y_scores = np.zeros((data_size), dtype=np.float32) 
    labels = []

    X1 = np.zeros((data_size, maxlen, 300), dtype=np.float32)
    X2 = np.zeros((data_size, maxlen, 300), dtype=np.float32)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(data_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = dict(zip(words.iterkeys(),xrange(len(words))))
    vocab["<UNK>"] = len(words) # Add unknown as word

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
                    idx = vocab["<UNK>"]

                else:
                    idx = vocab[toks_a[j-maxlen+len(toks_a)]] 

                X1[i, j] =  wordEmbeddings[:, idx]
                    
            for j in range(maxlen):
                if j < maxlen - len(toks_b):
                    idx = vocab["<UNK>"]
                else:
                    idx = vocab[toks_b[j-maxlen+len(toks_b)]]

                X2[i, j] =  wordEmbeddings[:, idx]


    Y_scores_pred = Y_scores_pred[:, 1:]
    Y_labels = np.zeros((len(labels), numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return X1, X2, Y_labels, Y_scores, Y_scores_pred


def read_dataset_tree(data_dir, name, rangeScores=5, numLabels=3, maxlen=36):

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

    vocab = dict(zip(words.iterkeys(),xrange(len(words))))
    vocab["<UNK>"] = len(words) # Add unknown as word

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

            """
            for j in range(maxlen):
                if j < maxlen - len(toks_a):
                    X1[i,j] = vocab["<UNK>"]
                else:
                    X1[i, j] = vocab[toks_a[j-maxlen+len(toks_a)]]
                    
            for j in range(maxlen):
                if j < maxlen - len(toks_b):
                    X2[i,j] = vocab["<UNK>"]
                else:
                    X2[i,j] = vocab[toks_b[j-maxlen+len(toks_b)]]
            """

    Y_scores_pred = Y_scores_pred[:, 1:]
    Y_labels = np.zeros((len(labels), numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return l_trees, r_trees, Y_labels, Y_scores, Y_scores_pred


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    read_dataset_tree(sick_dir, "train")
    read_dataset_tree(sick_dir, "dev")
    read_dataset_tree(sick_dir, "test")

