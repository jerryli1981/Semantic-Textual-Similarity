import numpy as np

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


def load_data_index(data, args, vocabSize):

    Y_scores_pred = np.zeros((len(data), args.rangeScores+1), dtype=np.float32)

    maxlen = 0
    for i, (label, score, l_t, r_t) in enumerate(data):

        max_ = max(len(l_t.nodes), len(r_t.nodes))
        if maxlen < max_:
            maxlen = max_

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
            if j < len(l_t.nodes):
                X1[i, j] =  l_t.nodes[j].index
            else:
                X1[i,j] = vocabSize-1

        for j in range(maxlen):
            if j < len(r_t.nodes):
                X2[i, j] =  r_t.nodes[j].index
            else:
                X2[i, j] = vocabSize-1

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
