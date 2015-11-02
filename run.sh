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
model=LSTM
optimizer=adagrad
mlpActivation=tanh #current only accept sigmoid and tanh
debug=False

outFile="models/${model}_wvecDim_${wvecDim}_step_${step}_optimizer_${optimizer}.bin"

python -u main.py --debug $debug --mlpActivation $mlpActivation --step $step --repModel $model \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim



