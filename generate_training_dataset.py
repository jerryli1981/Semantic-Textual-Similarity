import glob

_DEBUG = False
if _DEBUG == True:
	import pdb
	pdb.set_trace()

folders = ['sts2012', 'sts2013', 'sts2014']

with open("train.txt", "wb") as f:

	for folder in folders:

		gss = glob.glob("./"+folder+"/STS.gs."+"*.txt")
		inputs = glob.glob("./"+folder+"/STS.input."+"*.txt")

		for gs, ip in zip(gss,inputs):

			with open(gs, "rb") as f1, open(ip, "rb") as f2:

				for score, pair in zip(f1,f2):
					f.write(score.strip()+"\t"+pair.strip()+"\n")	 

