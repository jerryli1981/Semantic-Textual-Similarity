#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=200
step=1e-2
numLabels=5
wvecDim=100
numProcess=6
miniBatch=300

model="RNN" #either RNN, RNN2, RNN3, RNTN, or DCNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
outfile="models/${model}_wvecDim_${wvecDim}_step_${step}_multiP.bin"

echo $outfile

python -u runNNetMultiProcess.py --step $step --epochs $epochs --outFile $outfile \
                  				--outputDim $numLabels --minibatch $miniBatch --numProcess $numProcess --wvecDim $wvecDim --model $model 

