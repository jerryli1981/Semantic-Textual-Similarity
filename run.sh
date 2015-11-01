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
activation=tanh #current only accept sigmoid and tanh
debug=False


#outfile="models/${model}_wvecDim_${wvecDim}_step_${step}.bin"

python -u main.py --debug $debug --activation $activation --step $step --repModel $model \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs \
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim



