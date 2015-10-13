import glob
import re
import numpy as np
from stanford_parser_wrapper import Parser
import cPickle

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\(.*?\)", "", string)
    string = re.sub(r"\s{2,}", " ", string)       
    return string.strip()

def build_training_data():
    parser = Parser() 
    folders = ['sts2012', 'sts2013', 'sts2014']
    pairSet = set()   
    dataset = []
    index = 0
    for folder in folders:
        gss = glob.glob("./"+folder+"/STS.gs."+"*.txt")
        inputs = glob.glob("./"+folder+"/STS.input."+"*.txt")
        
        for gs, ip in zip(gss,inputs):
            with open(gs, "rb") as f1, open(ip, "rb") as f2:                            
                for score, pair in zip(f1,f2):
                    index += 1
                    if index % 200 == 0:
                        print index
                    items = pair.split('\t')
                    first_sent = items[0]
                    first_sent = clean_str(first_sent)
                    second_sent = items[1]
                    second_sent = clean_str(second_sent)
                    
                    if len(first_sent) ==0 or len(second_sent) ==0:
                        continue
                    if " " not in first_sent or " " not in second_sent:
                        continue

                    if pair not in pairSet:

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
            
                        datum = {   "score":score.strip(), 
                                    "text": (first_sent, second_sent), 
                                    "parse":(first_parse_output, second_parse_output),
                                    "split": np.random.randint(0,10)
                                }
                        dataset.append(datum)            
                        pairSet.add(pair)  
    
    with open("training_dataset","wb") as f:
        cPickle.dump(dataset,f)
   
if __name__ == "__main__":
    _debug_ = False
    if _debug_:
        import pdb
        pdb.set_trace()

    build_training_data()




