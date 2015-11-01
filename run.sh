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
wvecDim=300
miniBatch=200
model=LSTM
optimizer=adadelta
debug=False


######################################################## 
# Probably a good idea to let items below here be
########################################################
outfile="models/${model}_wvecDim_${wvecDim}_step_${step}.bin"

python -u main.py --debug $debug --step $step --repModel $model --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outfile \
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim



