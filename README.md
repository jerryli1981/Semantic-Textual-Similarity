# Semantic-Textual-Similarity

This is impelmentation of Tree-LSTM https://github.com/stanfordnlp/treelstm for Semantic Textual Similarity(STS) task.

Tree-LSTM is implemented by Torch. I would like to compare Torch with Theano. 

I try three solutions

1) main_lasagne.py is using Lasagne https://github.com/Lasagne/Lasagne.git

2) main_keras.py is using keras https://github.com/fchollet/keras.git

3) main_theano.py is pure theano implementation

Note that, it is hard to implement dynamic tree stucture LSTM via theano. So I leverage sequence LSTM that already
in the Lasagne and keras platforms to predict the relateness score. 

Known issue: both three solutions just get 70% pearson correlation which is so far away from 84% reported in their paper.
             If you are interested in fix this issue with me, please contact me: jerryli1981@gmail.com

