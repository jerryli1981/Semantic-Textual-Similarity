import cPickle
import numpy as np
import sys

def get_idx_from_sent(sent, word_idx_map, max_l=35, filter_h=6):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x
           
def generateFeatures_TreeRNN(revs, W, word_idx_map, fname, filter_h=6):
    import pandas as pd
    from RecursiveNeuralNetwork_Architecture import forward_prop
    from collections import defaultdict
       
    params, vocab, rel_dict, dim = cPickle.load(open("RNN_model_tree_"+fname,"rb"))

    max_nodes = np.max(pd.DataFrame(revs)["num_words_merged_Tree"])
    
    img_h = len(get_idx_from_sent(revs[0]["linearbyTree"], word_idx_map, max_nodes, filter_h))
    
    num_labels = np.max(pd.DataFrame(revs)["label"])+1
    
    pad = filter_h - 1
    datasets = []
    ids = []
    scores = []
    labels = []
    We = W[:, :dim]
    for rev in revs:    
        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])   
        tree = rev["tree"]
        tree.label = rev["label"]
        

        arr = []
        for i in xrange(pad):
          arr.append(We[0])
             
        for node in tree.get_nodes():             
                      
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]
                except:
                    node.ind = 0
            else:
                node.ind = 0 
                                
        for node in tree.get_nodes():
            node.vec = We[node.ind, :].reshape((dim, 1))
            
        forward_prop(params, tree, dim, num_labels)
    
        for node in tree.get_nodes(): 
            x = node.p_norm.reshape(dim)
            #x = node.vec.reshape(dim)
            arr.append(x)
  
                        
        while len(arr) < img_h:
            arr.append(We[0]) 
        

        datasets.append(np.asarray(arr).flatten()) 
        
                         
    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)
    
    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)
      
    return X, img_h, dim 

def generateFeatures_TreeWE(revs, W, word_idx_map, k=300, filter_h=6):
    """
    Transforms sentences into a 2-d matrix.
    """
    import pandas as pd
    max_l = np.max(pd.DataFrame(revs)["num_words_merged_Tree"])
    img_h = len(get_idx_from_sent(revs[0]["linearbyTree"], word_idx_map, max_l, filter_h))
    
    W = W[:, :k]
    
    ids = []
    scores = []
    labels = []
    datasets= []
    
    for rev in revs:
           
        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])
        
        sent_idx = get_idx_from_sent(rev["linearbyTree"], word_idx_map, max_l, filter_h) 
        
        arr = W[sent_idx] 

        datasets.append(np.asarray(arr).flatten()) 
                          
 
    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)

    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)
              
    return X, img_h, k   

def generateFeatures_GraphRNN(revs, W, word_idx_map, fname, filter_h=6):
    import pandas as pd
    from RecursiveNeuralNetwork_Architecture import forward_prop
    from collections import defaultdict
       
    params, vocab, rel_dict, dim = cPickle.load(open("RNN_model_graph_best","rb"))

    max_nodes = np.max(pd.DataFrame(revs)["num_words_merged_Graph"])
    img_h = len(get_idx_from_sent(revs[0]["linearbyGraph"], word_idx_map, max_nodes, filter_h))
   
    num_labels = np.max(pd.DataFrame(revs)["label"])+1
    
    pad = filter_h - 1
    datasets = []
    ids = []
    scores = []
    labels = []
    We = W[:, :dim]
    for rev in revs:    
        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])   
        graph = rev["graph"]
        graph.label = rev["label"]
        
        sent_idx = []

        arr = []
        for i in xrange(pad):
          arr.append(We[0])
             
        for node in graph.get_nodes():             
                      
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]
                except:
                    node.ind = 0
            else:
                node.ind = 0   
                                
        for node in graph.get_nodes():
            node.vec = We[node.ind, :].reshape((dim, 1))
            
        forward_prop(params, graph, dim, num_labels)
        
        for node in graph.get_nodes(): 
            x = node.p_norm.reshape(dim)
            #x = node.vec.reshape(dim)
            arr.append(x)
                         
        while len(arr) < img_h:
            arr.append(We[0]) 
        
        datasets.append(np.asarray(arr).flatten()) 
                               
    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)
    
    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)
      
    return X, img_h, dim 
    
    
def generateFeatures_GraphWE(revs, W, word_idx_map, k=300, filter_h=6):
    """
    Transforms sentences into a 2-d matrix.
    """
    import pandas as pd
    
    max_l = np.max(pd.DataFrame(revs)["num_words_merged_Graph"])
    img_h = len(get_idx_from_sent(revs[0]["linearbyGraph"], word_idx_map, max_l, filter_h))
    
    W = W[:, :k]
    
    ids = []
    scores = []
    labels = []
    
    datasets= []
    
    for rev in revs:
           
        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])
        
        sent_idx = get_idx_from_sent(rev["linearbyGraph"], word_idx_map, max_l, filter_h) 
        
        arr = W[sent_idx] 
        datasets.append(np.asarray(arr).flatten()) 
                          
 
    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)
    
    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)
              
    return X, img_h, k              

def generateFeatures_segment(revs, W, word_idx_map, k=300, filter_h=6, segmentSize = 3):

    import pandas as pd
    max_l = np.max(pd.DataFrame(revs)["num_words_single"])
    img_h = len(get_idx_from_sent(revs[0]["text"].split("\t")[0], word_idx_map, max_l, filter_h))
    
    W = W[:, :k]
    ids = []
    scores = []
    labels = []
    
    datasets= []
    
    for rev in revs:
        
        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])
        
        first_sent_idx = get_idx_from_sent(rev["text"].split("\t")[0], word_idx_map, max_l, filter_h)
        first_sent_segments = []        
        for i in range(len(first_sent_idx)-segmentSize+1):
            segment = [first_sent_idx[i+j] for j in range(segmentSize)]
            first_sent_segments.append(segment)
            
        second_sent_idx = get_idx_from_sent(rev["text"].split("\t")[1], word_idx_map, max_l, filter_h)
        second_sent_segments = []
        for i in range(len(second_sent_idx)-segmentSize+1):
            segment = [second_sent_idx[i+j] for j in range(segmentSize)]
            second_sent_segments.append(segment)
            
        arr = []
        intersection = set(first_sent_idx) & set(second_sent_idx)
        
        if len(intersection) != 0:
            for ele in intersection:
                arr.append(W[ele])
        else:
            for ele in set(first_sent_idx):
                arr.append(W[ele])
            
        """
        for fi, se in zip(first_sent_segments, second_sent_segments):
            ave = np.zeros(k)
            
            for ele in fi + se:
                    ave += W[ele]
                             
            arr.append(ave/6)
        """
              
        while len(arr) < img_h:
            arr.append(W[0]) 
            
        datasets.append(np.asarray(arr).flatten())        
                                               
    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)
    
    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)
            
    return X, img_h, k  

def generateFeaturess_TwoChannelsWE(revs, W, word_idx_map, k=300, filter_h=6):
    

    import pandas as pd
    max_l = np.max(pd.DataFrame(revs)["num_words_merged_Graph"]) #need choose the max
    img_h = len(get_idx_from_sent(revs[0]["linearbyGraph"].split("\t")[0], word_idx_map, max_l, filter_h))
    
    
    W = W[:, :k]
    ids = []
    scores = []
    labels = []
    
    datasets= []
    
    
    for rev in revs:

        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])
        
        first_sent_idx = get_idx_from_sent(rev["text"].split("\t")[0], word_idx_map, max_l, filter_h)          
        second_sent_idx = get_idx_from_sent(rev["text"].split("\t")[1], word_idx_map, max_l, filter_h)
       
        arr = []
        intersection = set(first_sent_idx) & set(second_sent_idx)
        
        if len(intersection) != 0:
            for ele in intersection:
                arr.append(W[ele])
        else:
            for ele in set(first_sent_idx):
                arr.append(W[ele])
              
        while len(arr) < img_h:
            arr.append(W[0]) 
            
        datasets.append(np.asarray(arr).flatten())        
                                             
    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)

    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)

 
    datasets= []

    for rev in revs:   

        sent_idx = get_idx_from_sent(rev["linearbyGraph"], word_idx_map, max_l, filter_h) 
        arr = W[sent_idx] 
        datasets.append(np.asarray(arr).flatten()) 
                          
 
    dataset_array_1 = np.asarray(datasets)    
    X1 = np.concatenate((dataset_array_1, id_array, score_array, label_array), axis=1)
      
    return X, X1, img_h, k 

def generateFeatures_TwoChannelsRNN(revs, W, word_idx_map, fname, filter_h=6):
    from RecursiveNeuralNetwork_Architecture import forward_prop
    import pandas as pd
    max_l = np.max(pd.DataFrame(revs)["num_words_merged_Graph"]) #need choose the max
    
    params, vocab, rel_dict, dim = cPickle.load(open("RNN_model_graph_"+fname,"rb"))
    num_labels = np.max(pd.DataFrame(revs)["label"])+1

    W = W[:, :dim]
    pad = filter_h - 1
    ids = []
    scores = []
    labels = []
    
    datasets= []
    
    img_h = len(get_idx_from_sent(revs[0]["linearbyGraph"].split("\t")[0], word_idx_map, max_l, filter_h))

    for rev in revs:
        
        graph = rev["graph"]
        graph.label = rev["label"]

        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])
        
        first_sent_idx = get_idx_from_sent(rev["text"].split("\t")[0], word_idx_map, max_l, filter_h)          
        second_sent_idx = get_idx_from_sent(rev["text"].split("\t")[1], word_idx_map, max_l, filter_h)
        intersection = set(first_sent_idx) & set(second_sent_idx)
        
        arr = []
        for i in xrange(pad):
          arr.append(W[0])
          
        
        for node in graph.get_nodes():             
                      
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]
                except:
                    node.ind = 0
            else:
                node.ind = 0   
                                
        for node in graph.get_nodes():
            node.vec = W[node.ind, :].reshape((dim, 1))
            
        forward_prop(params, graph, dim, num_labels)
        
        for node in graph.get_nodes(): 
            if node.ind in intersection:
                arr.append(node.p_norm.reshape(dim))
                   
        while len(arr) < img_h:
            arr.append(W[0]) 
            
        datasets.append(np.asarray(arr).flatten())        
                                             
    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)

    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)

    datasets= []

    for rev in revs:   

        graph = rev["graph"]
        graph.label = rev["label"]
        sent_idx = []

        arr = []
        for i in xrange(pad):
          arr.append(W[0])
             
        for node in graph.get_nodes():             
                      
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]
                except:
                    node.ind = 0
            else:
                node.ind = 0   
                                
        for node in graph.get_nodes():
            node.vec = W[node.ind, :].reshape((dim, 1))
            
        forward_prop(params, graph, dim, num_labels)
        
        for node in graph.get_nodes(): 
            arr.append(node.p_norm.reshape(dim))
  
                        
        while len(arr) < img_h:
            arr.append(W[0]) 
        
        datasets.append(np.asarray(arr).flatten()) 
                          
 
    dataset_array_1 = np.asarray(datasets)    
    X1 = np.concatenate((dataset_array_1, id_array, score_array, label_array), axis=1)
      
    return X, X1, img_h, dim

def generateFeatures_alignment(revs, W, word_idx_map, fname, filter_h=6):
    
    import pandas as pd
    from RecursiveNeuralNetwork_Architecture import forward_prop
    from collections import defaultdict
    import re
    import operator
    from scipy.spatial.distance import euclidean
    from collections import OrderedDict
           
    params, vocab, rel_dict, dim = cPickle.load(open("RNN_model_graph_best","rb"))
    max_nodes = np.max(pd.DataFrame(revs)["num_words_merged_Graph"])
    img_h = len(get_idx_from_sent(revs[0]["linearbyGraph"], word_idx_map, max_nodes, filter_h))
    
    num_labels = np.max(pd.DataFrame(revs)["label"])+1
    
    pad = filter_h - 1
    datasets = []
    ids = []
    scores = []
    labels = []
    We = W[:, :dim]
    c =0 
    
    for rev in revs:
        c += 1
        if c%1000 ==0:
            print c    
            
        ids.append(rev["id"])
        labels.append(rev["label"])
        scores.append(rev["score"])   
        
        graph = rev["graph"]
        graph.label = rev["label"]
        
        for node in graph.get_nodes():                      
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]                   
                except:
                    node.ind = 0
            else:
                node.ind = 0   
                                
        for node in graph.get_nodes():
            node.vec = We[node.ind, :].reshape((dim, 1))
            
        forward_prop(params, graph, dim, num_labels)
         
        id_vec_map = {}

        for node in graph.get_nodes(): 
            x = node.p_norm.reshape(dim)
            #x = node.vec.reshape(dim)
            id_vec_map[node.ind] = x
               
        alignment_distance_map = {}
        tp_vector_map = {}
        hp_vector_map = {}
        
        for tp, hp in rev["alignments"]:       
            tp_eles = re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',tp).groups()
            #tp_rel = tp_eles[0]
            tp_gov = tp_eles[1]
            tp_govIdx = int(tp_eles[2])
            tp_dep = tp_eles[3]
            tp_depIdx = int(tp_eles[4])
            
            hp_eles = re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',hp).groups()
            #hp_rel = hp_eles[0]
            hp_gov = hp_eles[1]
            hp_govIdx = int(hp_eles[2])
            hp_dep = hp_eles[3]
            hp_depIdx = int(hp_eles[4])
            
            if tp_gov == "ROOT":
                tp_gov_vector = We[0]
            else:
                tp_gov_vector = id_vec_map[word_idx_map[tp_gov]]
                #tp_gov_vector = We[word_idx_map[tp_gov]]

            tp_dep_vector = id_vec_map[word_idx_map[tp_dep]]
            #tp_dep_vector = We[word_idx_map[tp_dep]]

            tp_vector_map[tp] = (tp_gov_vector, tp_dep_vector)
            
            if hp_gov == "ROOT":
                hp_gov_vector = We[0]
            elif word_idx_map[hp_gov] in id_vec_map: 
                hp_gov_vector = id_vec_map[word_idx_map[hp_gov]]
            else:
                #print hp_gov, word_idx_map[hp_gov]
                hp_gov_vector = We[word_idx_map[hp_gov]]
            
            if word_idx_map[hp_dep] in id_vec_map:
                hp_dep_vector = id_vec_map[word_idx_map[hp_dep]]
                #hp_dep_vector = We[word_idx_map[hp_dep]]
            else:
                #print hp_dep, word_idx_map[hp_dep]
                hp_dep_vector = We[word_idx_map[hp_dep]]
      
            hp_vector_map[hp] = (hp_gov_vector, hp_dep_vector)
            
            tp_vector = np.concatenate((tp_gov_vector, tp_dep_vector))
            hp_vector = np.concatenate((hp_gov_vector, hp_dep_vector))
        
            alignment_distance_map[(tp, hp)] = euclidean(tp_vector, hp_vector)
        
             
        arr = []
        for i in xrange(pad):
          arr.append(We[0])

        for node in graph.get_nodes():
            targetTp = None
            for tp, hp in rev["alignments"]:       
                tp_eles = re.match(r'(\w+)\((.*)-(\d+), (.*)-(\d+).*',tp).groups()
                tp_gov = tp_eles[1]
                tp_govIdx = int(tp_eles[2])
                tp_dep = tp_eles[3]
                tp_depIdx = int(tp_eles[4])
                if node.word.lower() == tp_gov.lower():
                    targetTp = tp
                    break

            if targetTp != None:        
                (tp_gov_vector, tp_dep_vector) = tp_vector_map[targetTp]
            else:
                continue
        
            minDist = 1000
            targetHp = None
            for t, h in alignment_distance_map:
                if t == targetTp:
                    dist = alignment_distance_map[(t,h)]
                    if dist < minDist:
                        targetHp = h
                        minDist = dist
            if targetHp != None:        
                (hp_gov_vector, hp_dep_vector) = hp_vector_map[targetHp] 
            
            if len(arr) < img_h*2-5:

                #t_sub = np.subtract(tp_gov_vector,tp_dep_vector)
                #h_sub = np.subtract(hp_gov_vector,hp_dep_vector)
                #arr.append(t_sub)
                #arr.append(h_sub)

                arr.append(tp_gov_vector)
                arr.append(tp_dep_vector)
                arr.append(hp_gov_vector)
                arr.append(hp_dep_vector)

                #arr.append(np.concatenate((tp_gov_vector[0:75],tp_dep_vector[0:75],hp_gov_vector[:75], hp_dep_vector[0:75])))
                
                #arr.append(np.concatenate((tp_gov_vector[:150],tp_dep_vector[:150])))
                #arr.append(np.concatenate((hp_gov_vector[:150],hp_dep_vector[:150])))
      
        while len(arr) < img_h*2:
            arr.append(We[0]) 
                     
        datasets.append(np.asarray(arr).flatten())     

    label_array = np.asarray(labels,dtype = "int")[:,np.newaxis]
    id_array = np.asarray(ids,dtype = "int")[:,np.newaxis]
    score_array = np.asarray(scores,dtype = "float32")[:,np.newaxis]
    dataset_array = np.asarray(datasets)

    X = np.concatenate((dataset_array, id_array, score_array, label_array), axis=1)
                                  
    return X, img_h*2, dim

def trainingTreeRNN(revs, W, word_idx_map, vocab, num_folds, numepochs, fname, dim = 300):
    
    from DependencyTree import make_tree
    from RecursiveNeuralNetwork_Architecture import evaluate_DT_RNN
    from collections import defaultdict
    import pandas as pd
    partitions = []
    for i in range(num_folds):
        partition = []    
        for rev in revs:   
            if rev["split"] == -1:
                partition.append('Test')
            elif rev["split"]==i:            
                partition.append('Valid')        
            else:  
                partition.append('Train')
        partitions.append(partition) 
        
    newrevs = []      
    rel_dict = defaultdict(float)            
    for rev in revs:       
        tree =rev["tree"]
        tree.label = rev["label"]
        tree.score = rev["score"]
        for node in tree.get_nodes():             
            for ind, rel in node.kids:
                if rel not in rel_dict:
                    if rel != "root":
                        rel_dict[rel] += 1
                        
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]
                except:
                    node.ind = 0
            else:
                node.ind = 0
                        
        datum  = {"tree":tree}
        newrevs.append(datum)
    
    print "selecting the best partition"
    num_labels = np.max(pd.DataFrame(revs)["label"])+1
    
    test_results = []
    valid_results = []    
    best_test_perf = 0.
    
    for i in range(num_folds):
            partition = partitions[i];
            valid_perf, test_perf, params = evaluate_DT_RNN(newrevs, 
                                                    partition, 
                                                    W, 
                                                    rel_dict,
                                                    batch_size = 100,
                                                    n_labels = num_labels,
                                                    n_epochs=numepochs,
                                                    d = dim)
            
            print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
            test_results.append(test_perf)  
            valid_results.append(valid_perf)
            if(test_perf > best_test_perf):
                print "saving RNN model"
                cPickle.dump( ( params, vocab, rel_dict, dim), open('RNN_model_tree_'+fname, 'wb'))
                best_test_perf = test_perf  

                
    print('Average valid performance %f %%' %(np.mean(valid_results) * 100.))        
    print('Average test performance %f %%' %(np.mean(test_results) * 100.))    
    print('Best test performance %f %%' %(best_test_perf * 100.))          

    
def trainingGraphRNN(revs, W, word_idx_map, vocab, num_folds, numepochs, fname, dim = 300):
    
    from DependencyTree import make_tree
    from RecursiveNeuralNetwork_Architecture import evaluate_DT_RNN
    from collections import defaultdict
    import pandas as pd
    partitions = []
    for i in range(num_folds):
        partition = []    
        for rev in revs:   
            if rev["split"] == -1:
                partition.append('Test')
            elif rev["split"]==i:            
                partition.append('Valid')        
            else:  
                partition.append('Train')
        partitions.append(partition) 
        
    newrevs = []      
    rel_dict = defaultdict(float)       
    
         
    for rev in revs:       
        tree =rev["graph"]
        tree.label = rev["label"]
        tree.score = rev["score"]
        for node in tree.get_nodes():             
            for ind, rel in node.kids:
                if rel not in rel_dict:
                    if rel != "root":
                        rel_dict[rel] += 1
                          
            word = node.word.lower()
            if word != "root":
                try:
                    node.ind = word_idx_map[word]
                except:
                    node.ind = 0
            else:
                node.ind = 0
                        
        datum  = {"tree":tree}
        newrevs.append(datum)
    
    print "selecting the best partition"
    num_labels = np.max(pd.DataFrame(revs)["label"])+1
    test_results = []
    valid_results = []    
    best_test_perf = 0.
    
    for i in range(num_folds):
            partition = partitions[i];
            valid_perf, test_perf, params = evaluate_DT_RNN(newrevs, 
                                                    partition, 
                                                    W, 
                                                    rel_dict,
                                                    batch_size = 100,
                                                    n_labels = num_labels,
                                                    n_epochs=numepochs,
                                                    d = dim)
            
            print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
            test_results.append(test_perf)  
            valid_results.append(valid_perf)
            if(test_perf > best_test_perf):
                print "saving RNN model"
                cPickle.dump( ( params, vocab, rel_dict, dim), open('RNN_model_graph_'+fname, 'wb'))
                best_test_perf = test_perf  

                
    print('Average valid performance %f %%' %(np.mean(valid_results) * 100.))        
    print('Average test performance %f %%' %(np.mean(test_results) * 100.))    
    print('Best test performance %f %%' %(best_test_perf * 100.))         
  
                 
if __name__=="__main__":
    
    _DEBUG = False
    if _DEBUG == True:
        import pdb
        pdb.set_trace()

    import argparse
    
    parser = argparse.ArgumentParser(description='This is a script for text entailment')
    parser.add_argument('-m', '--mode', 
                        help='generateFeatures, evaluateModel', 
                        required = True)    
      
    parser.add_argument('-f', '--fname',
                        help='file name', 
                        required = True)
    
    parser.add_argument('-c', '--crossvalidation', 
                        help='the number of folds for cross validation', 
                        required = False)

    parser.add_argument('-n', '--numepochs',
                        help='the number of epochs for training', 
                        required = False)
      
    parser.add_argument('-t', '--task',
                        help='task:  SR or TE', 
                        required = False)
    
    parser.add_argument('-d', '--dimension',
                        help='dimension', 
                        required = False)
    
    parser.add_argument('-i', '--inputType',
                        help='Tree-RNN, Tree-WordEmbedding, Graph-RNN, Graph-WordEmbedding,' 
                        +'Segments, TwoChannel-WordEmbedding, TwoChannel-RNN, Alignments', 
                        required = False)
    
    parser.add_argument('-u', '--updatedRNN',
                        help='True or False', 
                        required = False)

    args= parser.parse_args()
    mode = args.mode
    fn = args.fname
       
    if mode=="generateFeatures":
        
        inputType = args.inputType
        if inputType == None:
            print "inputType can't be None"
            sys.exit()
            
            
        if args.dimension == None:
            print "Dimension can't be None"
            sys.exit()
            
        dimension = int(args.dimension) 
      
        print "loading data...",
        x = cPickle.load(open(fn,"rb"))
        revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
        print "data loaded!, Begin to generate features"
    
        anotherdatasets = None
        
        if inputType == "Tree-RNN":
                                      
            if args.updatedRNN == None:
                print "updatedRNN can't be None"
                sys.exit()
                
            if args.updatedRNN == "True":
                
                if args.crossvalidation == None:
                    print "-c, crossvalidation can't be None"
                    sys.exit()
                
                if args.numepochs == None:
                    print "-n, the number of epochs for training can't be None"
                    sys.exit()
                
                num_folds = int(args.crossvalidation)
                numepochs = int(args.numepochs)
                
                print "Begin to training DT-Tree-RNN"
                trainingTreeRNN(revs, W, word_idx_map, vocab, num_folds, numepochs, fn, dimension)
                print "training is done"
                
            elif args.updatedRNN == "False":
                print "Using trained RNN Model"
                 
            datasets, img_h, img_w = generateFeatures_TreeRNN(revs, 
                                                            W,
                                                            word_idx_map,
                                                            fn,
                                                            filter_h=6)
            
        elif inputType == "Tree-WordEmbedding":
           
            datasets, img_h, img_w = generateFeatures_TreeWE(revs, 
                                                            W, 
                                                            word_idx_map, 
                                                            k=dimension, 
                                                            filter_h=6)

        elif inputType == "Graph-RNN":
                                      
            if args.updatedRNN == None:
                print "updatedRNN can't be None"
                sys.exit()
            
            if args.updatedRNN == "True":
                
                if args.crossvalidation == None:
                    print "-c, crossvalidation can't be None"
                    sys.exit()
                
                if args.numepochs == None:
                    print "-n, the number of epochs for training can't be None"
                    sys.exit()
                
                num_folds = int(args.crossvalidation)
                numepochs = int(args.numepochs)
                
                print "Begin to training DT-Graph-RNN" 
                trainingGraphRNN(revs, W, word_idx_map, vocab, num_folds, numepochs, fn, dimension)
                print "training is done"
                
            if args.updatedRNN == "False":       
                print "Using trained RNN Model"
                
            datasets, img_h, img_w = generateFeatures_GraphRNN(revs, 
                                                               W, 
                                                               word_idx_map, 
                                                               fn,
                                                               filter_h=6)
        
        elif inputType == "Graph-WordEmbedding":
            
            datasets, img_h, img_w = generateFeatures_GraphWE(revs,  W,
            word_idx_map, k=dimension, filter_h=6)
            
        elif inputType == "Segments":               
            datasets, img_h, img_w = generateFeatures_segment(revs, 
                                                                W,
                                                                word_idx_map,
                                                                k=dimension,
                                                                filter_h=3)
        elif inputType == "TwoChannel-WordEmbedding":

            datasets, anotherdatasets, img_h, img_w = generateFeaturess_TwoChannelsWE(revs, 
                                                                W,
                                                                word_idx_map,
                                                                k=dimension,
                                                                filter_h=3)
            
        elif inputType == "TwoChannel-RNN":
            datasets, anotherdatasets, img_h, img_w = generateFeatures_TwoChannelsRNN(revs, 
                                                                W,
                                                                word_idx_map,
                                                                fn,
                                                                filter_h=3)
            
            
        elif inputType == "Alignments":
            
            if args.updatedRNN == None:
                print "updatedRNN can't be None"
                sys.exit()
            
            if args.updatedRNN == "True":
                
                if args.crossvalidation == None:
                    print "-c, crossvalidation can't be None"
                    sys.exit()
                
                if args.numepochs == None:
                    print "-n, the number of epochs for training can't be None"
                    sys.exit()
                
                num_folds = int(args.crossvalidation)
                numepochs = int(args.numepochs)

                print "Begin to training DT-Graph-RNN" 
                trainingGraphRNN(revs, W, word_idx_map, vocab, num_folds, numepochs, fn, dimension)
                print "training is done"
                
            if args.updatedRNN == "False":    
                print "Using trained RNN Model"
            
            
            print "begin to genreate alignments"
            datasets, img_h, img_w = generateFeatures_alignment(revs, 
                                                                W,
                                                                word_idx_map,
                                                                fn,
                                                                filter_h=6)
            

                     
        print "Begin to dump inputs"
        cPickle.dump([revs, datasets, img_w, img_h, anotherdatasets], open("datasets_"+fn, "wb"))
        print "inputs is dumped"    
        
    elif mode=="evaluateModel":
        from Convolutional_Architecture_MultiChannel import evaluate_conv_net_SR_MultiChannel,evaluate_conv_net_TE_MultiChannel
        from Convolutional_Architecture import evaluate_conv_net_SR, evaluate_conv_net_TE, evaluate_conv_net_TE_2
        task = args.task
        if task == None:
            print "Task can't be None"
            sys.exit()
            
        if args.crossvalidation == None:
                print "-c, crossvalidation can't be None"
                sys.exit()
                
        if args.numepochs == None:
                print "-n, the number of epochs for training can't be None"
                sys.exit()
                
        num_folds = int(args.crossvalidation)
        numepochs = int(args.numepochs)
         
        X = cPickle.load(open("datasets_"+fn,"rb"))
        revs = X[0]
        partitions = []
        
                
        print "Begin to generate partitons"
        for i in range(num_folds):
            partition = []    
            for rev in revs:   
                if rev["split"] == -1:
                    partition.append('Test')
                elif rev["split"]==i:            
                    partition.append('Valid')        
                else:  
                    partition.append('Train')
            partitions.append(partition)
          
                       
        datasets = X[1]
        img_w = X[2]
        img_h = X[3]
        anotherdatasets = None
        
        print "img_h ", img_h
        print "img_w ", img_w
        test_results = []
        valid_results = []    
        best_test_perf = 0.
        for i in range(num_folds):
            
            partition = partitions[i];

            if task == "SR":
                if anotherdatasets == None:
                    print "Run Single Channel"
                    valid_perf, test_perf = evaluate_conv_net_SR(datasets,
                                                              partition,
                                                              img_h,
                                                              batch_size=100,
                                                              lr_decay=0.95,
                                                              conv_non_linear="relu",
                                                              sqr_norm_lim=9, 
                                                              n_epochs=numepochs, 
                                                              layer_sizes=[100, 10, 9], 
                                                              filter_hs=[3,4,5],
                                                              img_w = img_w,
                                                              drop_out=True) 
                    print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
                    test_results.append(test_perf)  
                    valid_results.append(valid_perf)
                    if(test_perf > best_test_perf):
                        best_test_perf = test_perf
                else:
                    print "Run MultiChannel"
                    valid_perf, test_perf = evaluate_conv_net_SR_MultiChannel(datasets,
                                                              anotherdatasets,
                                                              partition,
                                                              img_h,
                                                              batch_size=100,
                                                              lr_decay=0.95,
                                                              conv_non_linear="relu",
                                                              sqr_norm_lim=9, 
                                                              n_epochs=numepochs, 
                                                              layer_sizes=[100, 10, 9], 
                                                              filter_hs=[3,4,5],
                                                              img_w = img_w,
                                                              drop_out=True) 
                    print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
                    test_results.append(test_perf)  
                    valid_results.append(valid_perf)
                    if(test_perf > best_test_perf):
                        best_test_perf = test_perf
            
            elif task == "TE":
                if anotherdatasets == None:
                    print "Run Single Channel"
                
                    valid_perf, test_perf = evaluate_conv_net_TE(datasets,
                                                                partition,
                                                                img_h,
                                                                batch_size=100,
                                                                lr_decay=0.95,
                                                                conv_non_linear="relu",
                                                                sqr_norm_lim=9, 
                                                                n_epochs=numepochs, 
                                                                layer_sizes=[100, 10, 3], 
                                                                filter_hs=[3,4,5],
                                                                img_w = img_w,
                                                                drop_out=True) 
                    """
                    valid_perf, test_perf = evaluate_conv_net_TE_2(datasets,
                                                                partition,
                                                                img_h,
                                                                batch_size=100,
                                                                lr_decay=0.95,
                                                                conv_non_linear="relu",
                                                                sqr_norm_lim=9, 
                                                                n_epochs=numepochs, 
                                                                layer_sizes=[10, 20, 10, 3], 
                                                                filter_hs=[3],
                                                                img_w = img_w,
                                                                drop_out=True) 
                                                                
                
                """
                
                else:
                    print "Run MultiChannel"
                    valid_perf, test_perf = evaluate_conv_net_TE_MultiChannel(datasets,
                                                              anotherdatasets,
                                                              partition,
                                                              img_h,
                                                              batch_size=100,
                                                              lr_decay=0.95,
                                                              conv_non_linear="relu",
                                                              sqr_norm_lim=9, 
                                                              n_epochs=numepochs, 
                                                              layer_sizes=[100, 10, 2], 
                                                              filter_hs=[3,4,5],
                                                              img_w = img_w,
                                                              drop_out=True) 
                    print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
                    test_results.append(test_perf)  
                    valid_results.append(valid_perf)
                    if(test_perf > best_test_perf):
                        best_test_perf = test_perf
                        
                print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
                test_results.append(test_perf)  
                valid_results.append(valid_perf)
                if(test_perf > best_test_perf):
                    best_test_perf = test_perf
        
                 
        print('Average valid performance %f %%' %(np.mean(valid_results) * 100.))        
        print('Average test performance %f %%' %(np.mean(test_results) * 100.))    
        print('Best test performance %f %%' %(best_test_perf * 100.))