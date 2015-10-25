#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=20000
step=1e-2
numLabels=5
wvecDim=100
miniBatch=300
model=RNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
outfile="models/${model}_wvecDim_${wvecDim}_step_${step}.bin"

echo $outfile

python -u main.py --step $step --epochs $epochs --outFile $outfile \
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim



