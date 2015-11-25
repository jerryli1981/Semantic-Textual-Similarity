import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath('../Lasagne'))

from lasagne.layers import InputLayer, LSTMLayer, NonlinearityLayer, SliceLayer, FlattenLayer, EmbeddingLayer,\
    ElemwiseMergeLayer, ReshapeLayer, get_output, get_all_params, get_output_shape, DropoutLayer,\
    DenseLayer,ElemwiseSumLayer,Conv2DLayer, Conv1DLayer, CustomRecurrentLayer, AbsSubLayer,\
    ConcatLayer, Pool1DLayer, FeaturePoolLayer,count_params,MaxPool2DLayer,MaxPool1DLayer

from lasagne.regularization import regularize_layer_params_weighted, l2, l1,regularize_layer_params,\
                                    regularize_network_params
from lasagne.nonlinearities import tanh, sigmoid, softmax, rectify
from lasagne.objectives import categorical_crossentropy, squared_error, categorical_accuracy
from lasagne.updates import sgd, adagrad, adadelta, nesterov_momentum, rmsprop, adam
from lasagne.init import GlorotUniform

from utils import read_sequence_dataset, iterate_minibatches,loadWord2VecMap

#double_lstm 67%
#single_lstm 64%
#2dconv CNN_Sentence 63% 
#lstm1dconv 62%
#lstm2dconv 60%
#Mymodel 35%

def build_network_single_lstm(args, input1_var, input1_mask_var, 
        input2_var, intut2_mask_var, target_var, wordEmbeddings, maxlen=36):

    print("Building model with single lstm")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    GRAD_CLIP = wordDim
    input_1 = InputLayer((None, maxlen),input_var=input1_var)
    batchsize, seqlen = input_1.input_var.shape
    input_1_mask = InputLayer((None, maxlen),input_var=input1_mask_var)
    emb_1 = EmbeddingLayer(input_1, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_1.params[emb_1.W].remove('trainable')

    lstm_1 = LSTMLayer(
        emb_1, num_units=args.lstmDim, mask_input=input_1_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)

    
    slice_1 = SliceLayer(lstm_1, indices=-1, axis=1) # out_shape (None, args.lstmDim)


    input_2 = InputLayer((None, maxlen),input_var=input2_var)
    input_2_mask = InputLayer((None, maxlen),input_var=input2_mask_var)
    emb_2 = EmbeddingLayer(input_2, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_2.params[emb_2.W].remove('trainable')

    lstm_2 = LSTMLayer(
        emb_2, num_units=args.lstmDim, mask_input=input_2_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)
 
    slice_2 = SliceLayer(lstm_2, indices=-1, axis=1)

    mul = ElemwiseMergeLayer([slice_1, slice_2], merge_function=T.mul)
    sub = AbsSubLayer([slice_1, slice_2], merge_function=T.sub)
    concat = ConcatLayer([mul, sub])

    hid = DenseLayer(concat, num_units=args.hiddenDim, nonlinearity=sigmoid)


    if args.task == "sts":
        network = DenseLayer(hid, num_units=5,nonlinearity=softmax)

    elif args.task == "ent":
        network = DenseLayer(hid, num_units=3,nonlinearity=softmax)

    lambda_val = 0.5 * 1e-4
    layers = {lstm_1:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)

    return network, penalty

def build_network_double_lstm(args, input1_var, input1_mask_var, 
        input2_var, intut2_mask_var, target_var, wordEmbeddings, maxlen=36):

    print("Building model with double lstm")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    GRAD_CLIP = wordDim
    input_1 = InputLayer((None, maxlen),input_var=input1_var)
    batchsize, seqlen = input_1.input_var.shape
    input_1_mask = InputLayer((None, maxlen),input_var=input1_mask_var)
    emb_1 = EmbeddingLayer(input_1, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_1.params[emb_1.W].remove('trainable')

    lstm_1 = LSTMLayer(
        emb_1, num_units=args.lstmDim, mask_input=input_1_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)

    lstm_1_back = LSTMLayer(
        emb_1, num_units=args.lstmDim, mask_input=input_1_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh, backwards=True)
    
    slice_1 = SliceLayer(lstm_1, indices=-1, axis=1) # out_shape (None, args.lstmDim)
    slice_1_back = SliceLayer(lstm_1_back, indices=0, axis=1) # out_shape (None, args.lstmDim)
    concat_1 = ConcatLayer([slice_1, slice_1_back])

    input_2 = InputLayer((None, maxlen),input_var=input2_var)
    input_2_mask = InputLayer((None, maxlen),input_var=input2_mask_var)
    emb_2 = EmbeddingLayer(input_2, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_2.params[emb_2.W].remove('trainable')

    lstm_2 = LSTMLayer(
        emb_2, num_units=args.lstmDim, mask_input=input_2_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)
 
    lstm_2_back = LSTMLayer(
        emb_2, num_units=args.lstmDim, mask_input=input_2_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh, backwards=True)
    

    slice_2 = SliceLayer(lstm_2, indices=-1, axis=1)
    slice_2_b = SliceLayer(lstm_2_back, indices=0, axis=1)
    concat_2 = ConcatLayer([slice_2, slice_2_b])
    

    mul = ElemwiseMergeLayer([concat_1, concat_2], merge_function=T.mul)
    sub = AbsSubLayer([concat_1, concat_2], merge_function=T.sub)
    concat = ConcatLayer([mul, sub])

    hid = DenseLayer(concat, num_units=args.hiddenDim, nonlinearity=sigmoid)

    if args.task == "sts":
        network = DenseLayer(hid, num_units=5,nonlinearity=softmax)

    elif args.task == "ent":
        network = DenseLayer(hid, num_units=3,nonlinearity=softmax)

    lambda_val = 0.5 * 1e-4
    layers = {lstm_1:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)

    return network, penalty

def build_network_lstm1dconv(args, input1_var, input1_mask_var, 
        input2_var, intut2_mask_var, target_var, wordEmbeddings, maxlen=36):

    
    print("Building model lstm + 1D Convolution")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    GRAD_CLIP = wordDim

    num_filters = 8
    filter_size = 9
    stride = 3 
    pool_size=2


    input_1 = InputLayer((None, maxlen),input_var=input1_var)
    batchsize, seqlen = input_1.input_var.shape

    input_1_mask = InputLayer((None, maxlen),input_var=input1_mask_var)
    emb_1 = EmbeddingLayer(input_1, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_1.params[emb_1.W].remove('trainable')

    lstm_1 = LSTMLayer(emb_1, num_units=args.lstmDim, mask_input=input_1_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)

    slice_1 = SliceLayer(lstm_1, indices=-1, axis=1) #(None, 150)

    reshape_1 = ReshapeLayer(slice_1, (batchsize, 1, args.lstmDim)) #(None, 1, 150)

    conv1d_1 = Conv1DLayer(reshape_1, num_filters=num_filters, filter_size=filter_size, #(None, 3, 48)
        stride=stride, nonlinearity=rectify,W=GlorotUniform())


    maxpool_1 = MaxPool1DLayer(conv1d_1, pool_size=pool_size) #(None, 3, 24)

    forward_1 = FlattenLayer(maxpool_1) #(None, 72)
  
    input_2 = InputLayer((None, maxlen),input_var=input2_var)
    input_2_mask = InputLayer((None, maxlen),input_var=input2_mask_var)
    emb_2 = EmbeddingLayer(input_2, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_2.params[emb_2.W].remove('trainable')

    lstm_2 = LSTMLayer(emb_2, num_units=args.lstmDim, mask_input=input_2_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)

    slice_2 = SliceLayer(lstm_2, indices=-1, axis=1)

    reshape_2 = ReshapeLayer(slice_2,(batchsize, 1, args.lstmDim))

    conv1d_2 = Conv1DLayer(reshape_2, num_filters=num_filters, filter_size=filter_size,
        stride=stride, nonlinearity=rectify,W=GlorotUniform())

    maxpool_2 = MaxPool1DLayer(conv1d_2, pool_size=pool_size) #(None, 3, 24)

    forward_2 = FlattenLayer(maxpool_2) #(None, 72)
 

    # elementwisemerge need fix the sequence length
    mul = ElemwiseMergeLayer([forward_1, forward_2], merge_function=T.mul)
    sub = AbsSubLayer([forward_1, forward_2], merge_function=T.sub)

    
    concat = ConcatLayer([mul, sub])
    hid = DenseLayer(concat, num_units=args.hiddenDim, nonlinearity=sigmoid)

    if args.task == "sts":
        network = DenseLayer(
                hid, num_units=5,
                nonlinearity=softmax)

    elif args.task == "ent":
        network = DenseLayer(
                hid, num_units=3,
                nonlinearity=softmax)

    lambda_val = 0.5 * 1e-4
    layers = {lstm_1:lambda_val, conv1d_1:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)

    return network, penalty

def build_network_lstm2dconv(args, input1_var, input1_mask_var, 
        input2_var, intut2_mask_var, target_var, wordEmbeddings, maxlen=36):

    
    print("Building model lstm + 2D Convolution")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    GRAD_CLIP = wordDim


    num_filters = 8
    filter_size=(2,9)
    stride = 1 
    pool_size=(1,2)

    input_1 = InputLayer((None, maxlen),input_var=input1_var)
    batchsize, seqlen = input_1.input_var.shape
    input_1_mask = InputLayer((None, maxlen),input_var=input1_mask_var)
    emb_1 = EmbeddingLayer(input_1, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_1.params[emb_1.W].remove('trainable')

    lstm_1 = LSTMLayer(
        emb_1, num_units=args.lstmDim, mask_input=input_1_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)

    lstm_1_back = LSTMLayer(
        emb_1, num_units=args.lstmDim, mask_input=input_1_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh, backwards=True)
    
    slice_1 = SliceLayer(lstm_1, indices=-1, axis=1) # out_shape (None, args.lstmDim)
    slice_1_back = SliceLayer(lstm_1_back, indices=0, axis=1) # out_shape (None, args.lstmDim)

    concat_1 = ConcatLayer([slice_1, slice_1_back], axis=1)


    reshape_1 = ReshapeLayer(concat_1, (batchsize, 1, 2, args.lstmDim))
    conv2d_1 = Conv2DLayer(reshape_1, num_filters=num_filters, filter_size=filter_size, stride=stride, #(None, 3, 1, 48)
        nonlinearity=rectify,W=GlorotUniform())
    maxpool_1 = MaxPool2DLayer(conv2d_1, pool_size=pool_size) #(None, 3, 1, 24)
    forward_1 = FlattenLayer(maxpool_1) #(None, 72)


    input_2 = InputLayer((None, maxlen),input_var=input2_var)
    input_2_mask = InputLayer((None, maxlen),input_var=input2_mask_var)
    emb_2 = EmbeddingLayer(input_2, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_2.params[emb_2.W].remove('trainable')

    lstm_2 = LSTMLayer(
        emb_2, num_units=args.lstmDim, mask_input=input_2_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)
 
    lstm_2_back = LSTMLayer(
        emb_2, num_units=args.lstmDim, mask_input=input_2_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh, backwards=True)
    

    slice_2 = SliceLayer(lstm_2, indices=-1, axis=1)
    slice_2_b = SliceLayer(lstm_2_back, indices=0, axis=1)
    concat_2 = ConcatLayer([slice_2, slice_2_b])

    reshape_2 = ReshapeLayer(concat_2, (batchsize, 1, 2, args.lstmDim))
    conv2d_2 = Conv2DLayer(reshape_2, num_filters=num_filters, filter_size=filter_size, stride=stride,
        nonlinearity=rectify,W=GlorotUniform())
    maxpool_2 = MaxPool2DLayer(conv2d_2, pool_size=pool_size)
    forward_2 = FlattenLayer(maxpool_2) #(None, 72)

 
    # elementwisemerge need fix the sequence length
    mul = ElemwiseMergeLayer([forward_1, forward_2], merge_function=T.mul)
    sub = AbsSubLayer([forward_1, forward_2], merge_function=T.sub)

    
    concat = ConcatLayer([mul, sub])
    hid = DenseLayer(concat, num_units=args.hiddenDim, nonlinearity=sigmoid)

    if args.task == "sts":
        network = DenseLayer(
                hid, num_units=5,
                nonlinearity=softmax)

    elif args.task == "ent":
        network = DenseLayer(
                hid, num_units=3,
                nonlinearity=softmax)

    lambda_val = 0.5 * 1e-4
    layers = {lstm_1:lambda_val, conv1d_1:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)

    return network, penalty

def build_network_2dconv(args, input1_var, input1_mask_var, 
        input2_var, intut2_mask_var, target_var, wordEmbeddings, maxlen=36):

    
    print("Building model with 2D Convolution")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]

    num_filters = 100
    
    stride = 1 

    #CNN_sentence config
    filter_size=(3, wordDim)
    pool_size=(maxlen-3+1,1)

    #two conv pool layer
    #filter_size=(10, 100)
    #pool_size=(4,4)

    input_1 = InputLayer((None, maxlen),input_var=input1_var)
    batchsize, seqlen = input_1.input_var.shape
    #input_1_mask = InputLayer((None, maxlen),input_var=input1_mask_var)
    emb_1 = EmbeddingLayer(input_1, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_1.params[emb_1.W].remove('trainable') #(batchsize, maxlen, wordDim)

    reshape_1 = ReshapeLayer(emb_1, (batchsize, 1, maxlen, wordDim))

    conv2d_1 = Conv2DLayer(reshape_1, num_filters=num_filters, filter_size=(filter_size), stride=stride, 
        nonlinearity=rectify,W=GlorotUniform()) #(None, 100, 34, 1)
    maxpool_1 = MaxPool2DLayer(conv2d_1, pool_size=pool_size) #(None, 100, 1, 1) 
  
    """
    filter_size_2=(4, 10)
    pool_size_2=(2,2)
    conv2d_1 = Conv2DLayer(maxpool_1, num_filters=num_filters, filter_size=filter_size_2, stride=stride, 
        nonlinearity=rectify,W=GlorotUniform()) #(None, 100, 34, 1)
    maxpool_1 = MaxPool2DLayer(conv2d_1, pool_size=pool_size_2) #(None, 100, 1, 1) (None, 100, 1, 20)
    """

    forward_1 = FlattenLayer(maxpool_1) #(None, 100) #(None, 50400)
 

    input_2 = InputLayer((None, maxlen),input_var=input2_var)
    #input_2_mask = InputLayer((None, maxlen),input_var=input2_mask_var)
    emb_2 = EmbeddingLayer(input_2, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_2.params[emb_2.W].remove('trainable')

    reshape_2 = ReshapeLayer(emb_2, (batchsize, 1, maxlen, wordDim))
    conv2d_2 = Conv2DLayer(reshape_2, num_filters=num_filters, filter_size=filter_size, stride=stride, 
        nonlinearity=rectify,W=GlorotUniform()) #(None, 100, 34, 1)
    maxpool_2 = MaxPool2DLayer(conv2d_2, pool_size=pool_size) #(None, 100, 1, 1)

    """
    conv2d_2 = Conv2DLayer(maxpool_2, num_filters=num_filters, filter_size=filter_size_2, stride=stride, 
        nonlinearity=rectify,W=GlorotUniform()) #(None, 100, 34, 1)
    maxpool_2 = MaxPool2DLayer(conv2d_2, pool_size=pool_size_2) #(None, 100, 1, 1)
    """

    forward_2 = FlattenLayer(maxpool_2) #(None, 100)

 
    # elementwisemerge need fix the sequence length
    mul = ElemwiseMergeLayer([forward_1, forward_2], merge_function=T.mul)
    sub = AbsSubLayer([forward_1, forward_2], merge_function=T.sub)
    concat = ConcatLayer([mul, sub])

    concat = ConcatLayer([forward_1,forward_2])

    hid = DenseLayer(concat, num_units=args.hiddenDim, nonlinearity=sigmoid)

    if args.task == "sts":
        network = DenseLayer(
                hid, num_units=5,
                nonlinearity=softmax)

    elif args.task == "ent":
        network = DenseLayer(
                hid, num_units=3,
                nonlinearity=softmax)

    prediction = get_output(network, {input_1:input1_var, input_2:input2_var})
    
    loss = T.mean(categorical_crossentropy(prediction,target_var))
    lambda_val = 0.5 * 1e-4

    layers = {conv2d_1:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)
    loss = loss + penalty


    params = get_all_params(network, trainable=True)

    if args.optimizer == "sgd":
        updates = sgd(loss, params, learning_rate=args.step)
    elif args.optimizer == "adagrad":
        updates = adagrad(loss, params, learning_rate=args.step)
    elif args.optimizer == "adadelta":
        updates = adadelta(loss, params, learning_rate=args.step)
    elif args.optimizer == "nesterov":
        updates = nesterov_momentum(loss, params, learning_rate=args.step)
    elif args.optimizer == "rms":
        updates = rmsprop(loss, params, learning_rate=args.step)
    elif args.optimizer == "adam":
        updates = adam(loss, params, learning_rate=args.step)
    else:
        raise "Need set optimizer correctly"
 

    test_prediction = get_output(network, deterministic=True)
    test_loss = T.mean(categorical_crossentropy(test_prediction,target_var))

    """
    train_fn = theano.function([input1_var, input1_mask_var, input2_var, intut2_mask_var, target_var], 
        loss, updates=updates, allow_input_downcast=True)
    """
    train_fn = theano.function([input1_var, input2_var, target_var], 
        loss, updates=updates, allow_input_downcast=True)

    if args.task == "sts":
        """
        val_fn = theano.function([input1_var, input1_mask_var, input2_var, intut2_mask_var, target_var], 
            [test_loss, test_prediction], allow_input_downcast=True)
        """
        val_fn = theano.function([input1_var, input2_var, target_var], 
            [test_loss, test_prediction], allow_input_downcast=True)

    elif args.task == "ent":
        #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
        test_acc = T.mean(categorical_accuracy(test_prediction, target_var))

        """
        val_fn = theano.function([input1_var, input1_mask_var, input2_var, intut2_mask_var, target_var], 
            [test_loss, test_acc], allow_input_downcast=True)
        """
        val_fn = theano.function([input1_var, input2_var, target_var], 
            [test_loss, test_acc], allow_input_downcast=True)

    return train_fn, val_fn

def build_network_MyModel(args, input1_var, input1_mask_var, 
        input2_var, intut2_mask_var, target_var, wordEmbeddings, maxlen=36):

    
    print("Building model LSTM + Featue Model + 2D Convolution +MLP")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    GRAD_CLIP = wordDim

    input_1 = InputLayer((None, maxlen),input_var=input1_var)
    batchsize, seqlen = input_1.input_var.shape
    input_1_mask = InputLayer((None, maxlen),input_var=input1_mask_var)
    emb_1 = EmbeddingLayer(input_1, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_1.params[emb_1.W].remove('trainable')
    lstm_1 = LSTMLayer(emb_1, num_units=args.lstmDim, mask_input=input_1_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh)

    input_2 = InputLayer((None, maxlen),input_var=input2_var)
    input_2_mask = InputLayer((None, maxlen),input_var=input2_mask_var)
    emb_2 = EmbeddingLayer(input_2, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb_2.params[emb_2.W].remove('trainable')
    lstm_2 = LSTMLayer(emb_2, num_units=args.lstmDim, mask_input=input_2_mask, grad_clipping=GRAD_CLIP,
        nonlinearity=tanh) 
 

    """
    concat = ConcatLayer([lstm_1, lstm_2],axis=0) #(None, 36, 150)

    concat = ConcatLayer([lstm_1, lstm_2],axis=1) #(None, 72, 150)
    """

    lstm_1 = SliceLayer(lstm_1, indices=slice(-6, None), axis=1)
    lstm_2 = SliceLayer(lstm_2, indices=slice(-6, None), axis=1)

    concat = ConcatLayer([lstm_1, lstm_2],axis=2) #(None, 36, 300)

    num_filters = 32
    stride = 1 
    """
    filter_size=(10, 10)
    pool_size=(4,4)
    """

    filter_size=(3, 10)
    pool_size=(2,2)

    reshape = ReshapeLayer(concat, (batchsize, 1, 6, 2*args.lstmDim))

    conv2d = Conv2DLayer(reshape, num_filters=num_filters, filter_size=filter_size,
            nonlinearity=rectify,W=GlorotUniform())
    
    maxpool = MaxPool2DLayer(conv2d, pool_size=pool_size) #(None, 32, 6, 72)


    """
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    conv2d = Conv2DLayer(maxpool, num_filters=32, filter_size=(5, 5),
            nonlinearity=rectify)
    maxpool = MaxPool2DLayer(conv2d, pool_size=(2, 2))

    """
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    hid = DenseLayer(DropoutLayer(maxpool, p=.2),num_units=128,nonlinearity=rectify)


    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    if args.task == "sts":
        network = DenseLayer(
                hid, num_units=5,
                nonlinearity=softmax)

    elif args.task == "ent":
        network = DenseLayer(
                hid, num_units=3,
                nonlinearity=softmax)

    lambda_val = 0.5 * 1e-4
    layers = {lstm_1:lambda_val, conv1d_1:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)

    return network, penalty

    
def generate_theano_func(args, network, penalty):

    prediction = get_output(network)
    #loss = T.mean(target_var * ( T.log(target_var + 1e-30) - T.log(prediction) ))
    loss = T.mean(categorical_crossentropy(prediction,target_var))
    #loss += 0.0001 * sum (T.sum(layer_params ** 2) for layer_params in get_all_params(network) )
    #penalty = sum ( T.sum(lstm_param**2) for lstm_param in lstm_params )
    #penalty = regularize_layer_params(l_forward_1_lstm, l2)
    #penalty = T.sum(lstm_param**2 for lstm_param in lstm_params)
    #penalty = 0.0001 * sum (T.sum(layer_params ** 2) for layer_params in get_all_params(l_forward_1) )

    loss = loss + penalty

    params = get_all_params(network, trainable=True)

    if args.optimizer == "sgd":
        updates = sgd(loss, params, learning_rate=args.step)
    elif args.optimizer == "adagrad":
        updates = adagrad(loss, params, learning_rate=args.step)
    elif args.optimizer == "adadelta":
        updates = adadelta(loss, params, learning_rate=args.step)
    elif args.optimizer == "nesterov":
        updates = nesterov_momentum(loss, params, learning_rate=args.step)
    elif args.optimizer == "rms":
        updates = rmsprop(loss, params, learning_rate=args.step)
    elif args.optimizer == "adam":
        updates = adam(loss, params, learning_rate=args.step)
    else:
        raise "Need set optimizer correctly"
 
    test_prediction = get_output(network, deterministic=True)
    test_loss = T.mean(categorical_crossentropy(test_prediction,target_var))

    train_fn = theano.function([input1_var, input1_mask_var, input2_var, input2_mask_var, target_var], 
        loss, updates=updates, allow_input_downcast=True)

    if args.task == "sts":
        val_fn = theano.function([input1_var, input1_mask_var, input2_var, input2_mask_var, target_var], 
            [test_loss, test_prediction], allow_input_downcast=True)

    elif args.task == "ent":
        #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
        test_acc = T.mean(categorical_accuracy(test_prediction, target_var))
        val_fn = theano.function([input1_var, input1_mask_var, input2_var, input2_mask_var, target_var], 
            [test_loss, test_acc], allow_input_downcast=True)

    return train_fn, val_fn


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Usage")
    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=20)
    parser.add_argument("--step",dest="step",type=float,default=0.01)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--lstmDim",dest="lstmDim",type=int,default=30)
    parser.add_argument("--task",dest="task",type=str,default=None)
    args = parser.parse_args()

    print("Loading data...")
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    X1_train, X1_mask_train, X2_train, X2_mask_train, Y_labels_train, Y_scores_train, Y_scores_pred_train = \
        read_sequence_dataset(sick_dir, "train")
    X1_dev, X1_mask_dev, X2_dev, X2_mask_dev, Y_labels_dev, Y_scores_dev, Y_scores_pred_dev = \
        read_sequence_dataset(sick_dir, "dev")
    X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, Y_scores_test, Y_scores_pred_test = \
        read_sequence_dataset(sick_dir, "test")

    input1_var = T.imatrix('inputs_1')
    input2_var = T.imatrix('inputs_2')
    input1_mask_var = T.matrix('inputs_mask_1')
    input2_mask_var = T.matrix('inputs_mask_2')
    target_var = T.fmatrix('targets')

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings.astype(np.float32)

    """
    network, penalty= build_network_single_lstm(args, input1_var, input1_mask_var, input2_var, input2_mask_var,
        target_var, wordEmbeddings)
    train_fn, val_fn = generate_theano_func(args, network, penalty)
    """

    
    train_fn, val_fn = build_network_2dconv(args, input1_var, input1_mask_var, input2_var, input2_mask_var,
        target_var, wordEmbeddings)
    

    print("Starting training...")
    best_val_acc = 0
    best_val_pearson = 0
    for epoch in range(args.epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X1_train, X1_mask_train, X2_train, X2_mask_train, Y_labels_train,
            Y_scores_train, Y_scores_pred_train, args.minibatch, shuffle=True):

            inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

            if args.task == "sts":
                #train_err += train_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, scores_pred)
                train_err += train_fn(inputs1, inputs2, scores_pred)
            elif args.task == "ent":
                train_err += train_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, labels)
                #train_err += train_fn(inputs1, inputs2, labels)
            else:
                raise "task need to be set"

            train_batches += 1
 
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_pearson = 0

        for batch in iterate_minibatches(X1_dev, X1_mask_dev, X2_dev, X2_mask_dev, Y_labels_dev, Y_scores_dev, 
            Y_scores_pred_dev, len(X1_dev), shuffle=False):

            inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

            if args.task == "sts":

                #err, preds = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, scores_pred)
                err, preds = val_fn(inputs1, inputs2, scores_pred)
                predictScores = preds.dot(np.array([1,2,3,4,5]))
                guesses = predictScores.tolist()
                scores = scores.tolist()
                pearson_score = pearsonr(scores,guesses)[0]
                val_pearson += pearson_score 


            elif args.task == "ent":
                err, acc = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, labels)
                #err, acc = val_fn(inputs1, inputs2, labels)
                val_acc += acc

            val_err += err
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        if args.task == "sts":
            val_score = val_pearson / val_batches * 100
            print("  validation pearson:\t\t{:.2f} %".format(
                val_pearson / val_batches * 100))
            if best_val_pearson < val_score:
                best_val_pearson = val_score

        elif args.task == "ent":
            val_score = val_acc / val_batches * 100
            print("  validation accuracy:\t\t{:.2f} %".format(val_score))
            if best_val_acc < val_score:
                best_val_acc = val_score

    test_err = 0
    test_acc = 0
    test_pearson = 0
    test_batches = 0
    for batch in iterate_minibatches(X1_test, X1_mask_test, X2_test, X2_mask_test, Y_labels_test, 
        Y_scores_test, Y_scores_pred_test, len(X1_test), shuffle=False):

        inputs1, inputs1_mask, inputs2, inputs2_mask, labels, scores, scores_pred = batch

        if args.task == "sts":

            #err, preds = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, scores_pred)
            err, preds = val_fn(inputs1, inputs2, scores_pred)
            predictScores = preds.dot(np.array([1,2,3,4,5]))
            guesses = predictScores.tolist()
            scores = scores.tolist()
            pearson_score = pearsonr(scores,guesses)[0]
            test_pearson += pearson_score 

        elif args.task == "ent":
            err, acc = val_fn(inputs1, inputs1_mask, inputs2, inputs2_mask, labels)
            #err, acc = val_fn(inputs1, inputs2, labels)
            test_acc += acc

        test_err += err 
        test_batches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    if args.task == "sts":
        print("  Best validate perason:\t\t{:.2f} %".format(best_val_pearson))
        print("  test pearson:\t\t{:.2f} %".format(
            test_pearson / test_batches * 100))
    elif args.task == "ent":
        print("  Best validate accuracy:\t\t{:.2f} %".format(best_val_acc))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))