"""
Preprocessing script for SICK data.

"""

import os
import glob

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)

def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
        % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
         open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile, \
         open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
         open(os.path.join(dst_dir, 'sim.txt'), 'w') as simfile,\
         open(os.path.join(dst_dir, 'label.txt'),'w') as labelfile:
            datafile.readline()
            for line in datafile:
                i, a, b, sim, ent = line.strip().split('\t')
                idfile.write(i+'\n')
                afile.write(a+'\n')
                bfile.write(b+'\n')
                simfile.write(sim+'\n')
                labelfile.write(ent+'\n')

def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)
    dependency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=True)
    constituency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)
    constituency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=True)


def build_word2Vector(glove_path, sick_dir, vocab_name):

    print "building word2vec"
    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(sick_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = dict(zip(words.iterkeys(),xrange(len(words))))
    vocab["<UNK>"] = len(words) # Add unknown as word

    import gzip
    import numpy as np
    word_vecs = {}
    with gzip.open(glove_path, "rb") as f:
        for line in f:
           toks = line.split(' ')
           word = toks[0]
           if word in vocab:
               word_vecs[word] = np.fromiter(toks[1:], dtype='float32')  
               
    k = 300
    min_df = 1
    
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

    assert len(vocab) == len(word_vecs), "length of vocab mush equal with word_vecs"   

    vocab_size = len(word_vecs)
    word_embedding_matrix = np.zeros(shape=(k,vocab_size))            

    for i, word in enumerate(word_vecs):
        word_embedding_matrix[:,i] = word_vecs[word]

    import cPickle as pickle
    with open(os.path.join(sick_dir, 'word2vec.bin'),'w') as fid:
        pickle.dump(word_embedding_matrix,fid)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing SICK dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(sick_dir, 'train')
    dev_dir = os.path.join(sick_dir, 'dev')
    test_dir = os.path.join(sick_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    # split into separate files
    split(os.path.join(sick_dir, 'SICK_train.txt'), train_dir)
    split(os.path.join(sick_dir, 'SICK_trial.txt'), dev_dir)
    split(os.path.join(sick_dir, 'SICK_test_annotated.txt'), test_dir)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(dev_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab.txt'))
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab-cased.txt'),
        lowercase=False)
    
    glove_path = os.path.join(base_dir, 'glove.6B.300d.txt.gz')
    vocab_path = os.path.join(sick_dir, 'vocab-cased.txt')
    build_word2Vector(glove_path, sick_dir, 'vocab-cased.txt')
