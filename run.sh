#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=100
step=0.001
numLabels=3
rangeScores=5
hiddenDim=50
wvecDim=200
miniBatch=25
mlpActivation=sigmoid
optimizer=adam
task=sts


export THEANO_FLAGS=mode=FAST_RUN,device=$2,floatX=float32

if [ "$1" == "keras" ]
then
echo "run keras"
python -u main_keras.py --task $task --step $step --mlpActivation $mlpActivation \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs \
                  			--rangeScores $rangeScores	--numLabels $numLabels\
                  			--minibatch $miniBatch --wvecDim $wvecDim

elif [ "$1" == "lasagne" ]
then
echo "run lasagne"
python -u main_lasagne.py --task $task --step $step --mlpActivation $mlpActivation \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs \
                  			--rangeScores $rangeScores	--numLabels $numLabels\
                  			--minibatch $miniBatch --wvecDim $wvecDim
fi





