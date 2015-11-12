#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=20
step=0.01
numLabels=5
hiddenDim=100
wvecDim=200
miniBatch=128
model=LSTM
optimizer=adagrad
debug=False
useLearnedModel=False


outFile="models/${model}_wvecDim_${wvecDim}_step_${step}_optimizer_${optimizer}.bin"

#
#python -u main.py --debug $debug --useLearnedModel $useLearnedModel --step $step --repModel $model \
#				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
#                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim


if [ "$1" == "keras" ]
then
echo "run rnn_mlp_keras"
python -u rnn_mlp_keras.py --debug $debug --useLearnedModel $useLearnedModel --step $step --repModel $model \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim

elif [ "$1" == "lasagne" ]
then
echo "run rnn_mlp_lasagne"
python -u rnn_mlp_lasagne.py --debug $debug --useLearnedModel $useLearnedModel --step $step --repModel $model \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim

elif [ "$1" == "theano" ]
then
echo "run rnn_mlp_theano"
python -u rnn_mlp_theano.py --debug $debug --useLearnedModel $useLearnedModel --step $step --repModel $model \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim

fi





