import numpy as np
import gzip
import sys
import cPickle
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf8')


def get_word_matrix(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1 # it is make sence because i=0 is zero vector as padding
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def build_word2Vector_glove(fname, vocab):
    """
    Loads 300x1 word vecs from glove
    """
    word_vecs = {}
    with gzip.open(fname, "rb") as f:
        for line in f:
           toks = line.split(' ')
           word = toks[0]
           if word in vocab:
               word_vecs[word] = np.fromiter(toks[1:], dtype='float32')  
               
    k = 300
    min_df = 1
    
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

    
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    word_embedding_matrix = np.zeros(shape=(vocab_size+1, k))            
    word_embedding_matrix[0] = np.zeros(k)
    i = 1 # it is make sence because i=0 is zero vector as padding
    for word in word_vecs:
        word_embedding_matrix[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
        
    return word_embedding_matrix, word_idx_map

def build_word2Vector_mikolov(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    k = 300
    min_df = 1
    word_vecs = {}
    with gzip.open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
                
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
            
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    word_embedding_matrix = np.zeros(shape=(vocab_size+1, k))            
    word_embedding_matrix[0] = np.zeros(k)
    i = 1 # it is make sence because i=0 is zero vector as padding
    for word in word_vecs:
        word_embedding_matrix[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
        
    return word_embedding_matrix, word_idx_map

  
def build_training_data(data_file_path, cv=10):
    from stanford_parser_wrapper import Parser
    import traceback
    parser = Parser()    
    revs = []
    vocab = defaultdict(float)

    with open(data_file_path, "rb") as f:
        
        for index, line in enumerate(f):
            if index % 100 == 0:   
                print index  

            instance = line.strip().split('\t')  
            first_sentence = instance[0].strip()
            second_sentence = instance[1].strip()
                    
            try:
                first_parse_output = parser.parseSentence(first_sentence)
            except:
                print "first_sentence can't be parsing"
                #print first_sentence
                #traceback.print_exc()
                continue
            try:
                second_parse_output = parser.parseSentence(second_sentence)
            except:
                print "second_sentence can't be parsing"
                #print second_sentence
                #traceback.print_exc()
                continue
                                        
            score = instance[2].strip()
                                                    
            for word in first_parse_output["tokens"]:
                vocab[word.lower()] += 1
                
            for word in second_parse_output["tokens"]:
                vocab[word.lower()] += 1

            datum  = {"score":score, 
                      "text": (first_sentence, second_sentence), 
                      "parse":(first_parse_output, second_parse_output),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
            
    return revs, vocab
           
if  __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='This is a script for prepprocessing datasets')
    
    parser.add_argument('-i', metavar='--inputFilePath', 
            			help='input file path', 
                        required = True)
    
    parser.add_argument('-m', metavar ='--mode',
                        help='training or testing', 
                        required = True)
    
    parser.add_argument('-o', metavar='--outputFileName', 
                        help='the output file name', 
                        required = True)
    
    parser.add_argument('-c', metavar='--crossvalidation', 
                        help='the number of folds for cross validation', 
                        required = False)
    
    parser.add_argument('-k', metavar='--mikolov', 
                        help='the path of mikolov word2vector', 
                        required = False)
    
    parser.add_argument('-g', metavar='--glove', 
                        help='the path of glove word2vector', 
                        required = False)
    parser.add_argument('-d', metavar='--debug',
                        help='set debug mode, value is debug',
                        required = False)
    
    args= parser.parse_args()
    if args.d == "debug":
	import pdb
	pdb.set_trace()

    if args.k == None and args.g == None:
        print "word2vec can't be None"
        sys.exit()

    w2v_file_m = args.k
    w2v_file_g = args.g

    data_file_path = args.i
    output_file_path = args.o

    if args.m == "training":
        if args.c == None:
            print "crossvalidation can't be None"
            sys.exit()

        num_folds = int(args.c)
        
        print "Loading data...",        
        revs, vocab = build_training_data(data_file_path, cv=num_folds)
        print "data loaded!"
        
        print "number of instances: " + str(len(revs))
        print "vocab size: " + str(len(vocab))
        
        print "building word2vec from trained word embedding"
        if w2v_file_m != None:
            word_embedding_matrix, word_idx_map = build_word2Vector_mikolov(w2v_file_m, vocab)
        elif w2v_file_g != None:
            word_embedding_matrix, word_idx_map = build_word2Vector_glove(w2v_file_g, vocab) 
        print "word2vec is builded"
    
        cPickle.dump([revs, word_embedding_matrix, word_idx_map, vocab], open(output_file_path, "wb"))
        print "datasets created!"
    
