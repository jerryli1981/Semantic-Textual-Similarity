#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=100
step=0.001
hiddenDim=50
lstmDim=150
miniBatch=25
optimizer=adam

export THEANO_FLAGS=mode=FAST_RUN,device=$1,floatX=float32

if [ "$2" == "keras" ]
then
echo "run keras"

task=$3

python -u main_keras.py --task $task --step $step \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs \
                  			--minibatch $miniBatch --lstmDim $lstmDim

elif [ "$2" == "lasagne" ]
then
echo "run lasagne"

task=$3

python -u main_lasagne.py --task $task --step $step \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs \
                  			--minibatch $miniBatch --lstmDim $lstmDim

elif [ "$2" == "theano" ]
then
echo "run theano"
python -u main_theano.py --step $step --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs \
                  			--minibatch $miniBatch --lstmDim $lstmDim

fi





