import glob
import re


_DEBUG = False
if _DEBUG == True:
	import pdb
	pdb.set_trace()

folders = ['sts2012', 'sts2013', 'sts2014']

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\(.*?\)", "", string)
	string = re.sub(r"\s{2,}", " ", string)       
	return string.strip()

pairSet = set()

with open("train.txt", "wb") as f:

		for folder in folders:

			gss = glob.glob("./"+folder+"/STS.gs."+"*.txt")
			inputs = glob.glob("./"+folder+"/STS.input."+"*.txt")

			for gs, ip in zip(gss,inputs):

				with open(gs, "rb") as f1, open(ip, "rb") as f2:
                                        
					for score, pair in zip(f1,f2):
						
                                                items = pair.split('\t')
                                                first_sent = items[0]
						first_sent = clean_str(first_sent)
                                                second_sent = items[1]
						second_sent = clean_str(second_sent)
                                            
						if pair not in pairSet:
							f.write(first_sent+"\t"+second_sent+"\t"+score.strip()+"\n")
							pairSet.add(pair)	 
	
