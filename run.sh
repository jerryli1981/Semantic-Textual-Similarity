#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=1000
step=0.01
numLabels=5
hiddenDim=50
wvecDim=100
miniBatch=300
model=RNN
optimizer=adadelta


######################################################## 
# Probably a good idea to let items below here be
########################################################
outfile="models/${model}_wvecDim_${wvecDim}_step_${step}.bin"

echo $outfile

python -u main.py --step $step --repModel $model --hiddenDim $hiddenDim --epochs $epochs --outFile $outfile \
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim



