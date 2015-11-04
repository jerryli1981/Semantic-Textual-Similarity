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
miniBatch=200
model=RNN
optimizer=adagrad
debug=False
useLearnedModel=True

outFile="models/${model}_wvecDim_${wvecDim}_step_${step}_optimizer_${optimizer}.bin"

#
#python -u main.py --debug $debug --useLearnedModel $useLearnedModel --step $step --repModel $model \
#				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
#                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim


python -u main_theano.py --debug $debug --useLearnedModel $useLearnedModel --step $step --repModel $model \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim



