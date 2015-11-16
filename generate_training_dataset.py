import glob
import re
from stanford_parser_wrapper import Parser
import cPickle
import os
import dependency_tree as dt

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

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

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\(.*?\)", "", string)
    string = re.sub(r"\s{2,}", " ", string)       
    return string.strip()

def build_datasets():
    parser = Parser() 
    folders = ['train', 'dev', 'test']
    
    for folder in folders:
        index = 0
        dataset = []
        a_s = "./data/sick/"+folder+"/a.txt"
        b_s = "./data/sick/"+folder+"/b.txt"
        sims = "./data/sick/"+folder+"/sim.txt"
        labs = "./data/sick/"+folder+"/label.txt"        

        with open(a_s, "rb") as f1, \
             open(b_s, "rb") as f2, \
             open(sims, "rb") as f3, \
             open(labs, 'rb') as f4:
                            
            for a, b, sim, ent in zip(f1,f2,f3,f4):
                index += 1
                if index % 200 == 0:
                    print index
                
                first_sent = clean_str(a)
                second_sent = clean_str(b)
                
                if len(first_sent) ==0 or len(second_sent) ==0:
                    continue
                if " " not in first_sent or " " not in second_sent:
                    continue

                try:
                    first_parse_output = parser.parseSentence(first_sent)
                except:
                    print "first_sentence can't be parsing"
                    #print first_sentence
                    #traceback.print_exc()
                    continue
                try:
                    second_parse_output = parser.parseSentence(second_sent)
                except:
                    print "second_sentence can't be parsing"
                    #print second_sentence
                    #traceback.print_exc()
                    continue
    
                datum = {   "score":sim.strip(),
                            "label":ent.strip(), 
                            "text": (first_sent, second_sent), 
                            "parse":(first_parse_output, second_parse_output)
                        }
                dataset.append(datum)            

        with open(folder+"_dataset","wb") as f:
            cPickle.dump(dataset,f)

if __name__ == "__main__":
    print("=" * 80)
    print "Preprocessing SICK dataset"
    print("=" * 80)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')
    train_dir = os.path.join(sick_dir, 'train')
    dev_dir = os.path.join(sick_dir, 'dev')
    test_dir = os.path.join(sick_dir,'test')
    make_dirs([train_dir, dev_dir, test_dir])

    split(os.path.join(sick_dir, 'SICK_train.txt'), train_dir)
    split(os.path.join(sick_dir, 'SICK_trial.txt'), dev_dir)
    split(os.path.join(sick_dir, 'SICK_test_annotated.txt'), test_dir)

    build_datasets()

    dt.buildWordRelMap("train_dataset","dev_dataset", "test_dataset")

    dt.build_word2Vector_glove()


