#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/18/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from RNN import RNN


# load the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)


# # Part I. Vinalla RNN with Sequence of Pixels

# define parameters
category = 'basicRNN'
learning_rate = 0.0005
max_iters = 2000
batch_size = 100
n_input = 1
n_steps = 784
n_hidden = 100
n_classes = 10
regression = 'linear'

# build the model
model = RNN(category, learning_rate, max_iters, batch_size, 
            n_input, n_steps, n_hidden, n_classes, regression)

# begin training
model.train(mnist, order='C', frequence=1, size=None, show_frequence=100)

# get training accuracy and testing accuracy
model1_train_acc, model1_cv_acc = model.get_params()

# plot the training and testing curve
fig, ax = plt.subplots()
ax.plot(range(1, len(model1_train_acc)+1), model1_train_acc, 'g-', label='Training Accuracy')
ax.plot(range(1, len(model1_cv_acc)+1), model1_cv_acc, 'r-', label='Validation Accuracy')
ax.legend(loc=4, fontsize=10)
plt.show()


# # Part II. LSTM with Sequence of Pixels

# define parameters
category = 'LSTM'
learning_rate = 0.0001
max_iters = 1000
batch_size = 100
n_input = 1
n_steps = 784
n_hidden = 100
n_classes = 10
regression = 'linear'

# build the model
model = RNN(category, learning_rate, max_iters, batch_size, 
            n_input, n_steps, n_hidden, n_classes, regression)

# begin training
model.train(mnist, order='C', frequence=1, size=None, show_frequence=50)

# get training accuracy and testing accuracy
model2_train_acc, model2_cv_acc = model.get_params()

# plot the training and testing curve
fig, ax = plt.subplots()
ax.plot(range(1, len(model2_train_acc)+1), model2_train_acc, 'g-', label='Training Accuracy')
ax.plot(range(1, len(model2_cv_acc)+1), model2_cv_acc, 'r-', label='Validation Accuracy')
ax.legend(loc=4, fontsize=10)
plt.show()


# # Part III. Vinalla RNN with Sequence of Columns

# define parameters
# 28 steps, inputs is 28 column vector
category = 'basicRNN'
learning_rate = 0.0001
max_iters = 5000
batch_size = 100
n_input = 28
n_steps = 28
n_hidden = 100
n_classes = 10
regression = 'linear'

# build the model
model = RNN(category, learning_rate, max_iters, batch_size, 
            n_input, n_steps, n_hidden, n_classes, regression)

# begin training
model.train(mnist, order='F', frequence=1, size=None, show_frequence=500)

# get training accuracy and testing accuracy
model3_train_acc, model3_cv_acc = model.get_params()

# plot the training and testing curve
fig, ax = plt.subplots()
ax.plot(range(1, len(model3_train_acc)+1), model3_train_acc, 'g-', label='Training Accuracy')
ax.plot(range(1, len(model3_cv_acc)+1), model3_cv_acc, 'r-', label='Validation Accuracy')
ax.legend(loc=4, fontsize=10)
plt.show()


# # Part IV. LSTM with Sequence of Columns

# define parameters 
# 28 steps, inputs is 28 column vector
category = 'LSTM'
learning_rate = 0.0001
max_iters = 5000
batch_size = 100
n_input = 28
n_steps = 28
n_hidden = 100
n_classes = 10
regression = 'linear'

# build the model
model = RNN(category, learning_rate, max_iters, batch_size, 
            n_input, n_steps, n_hidden, n_classes, regression)

# begin training
model.train(mnist, order='F', frequence=1, size=None, show_frequence=500)

# get training accuracy and testing accuracy
model4_train_acc, model4_cv_acc = model.get_params()

# plot the training and testing curve
fig, ax = plt.subplots()
ax.plot(range(1, len(model4_train_acc)+1), model4_train_acc, 'g-', label='Training Accuracy')
ax.plot(range(1, len(model4_cv_acc)+1), model4_cv_acc, 'r-', label='Validation Accuracy')
ax.legend(loc=4, fontsize=10)
plt.show()

# plot the convergence curve
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
plt.tight_layout(pad=3.0, h_pad=3.5, w_pad=1.0, rect=None)

ax[0, 0].plot(range(1, len(model1_train_acc)+1), model1_train_acc, 
              'g-', label='Training Acc')
ax[0, 0].set_ylim([0, 0.4])
ax[0, 0].set_title('Basic RNN with 784 steps')
ax[0, 0].set_xlabel('Training Iterations', fontsize=9)
ax[0, 0].set_ylabel('Accuracy')
ax[0, 0].legend(loc='best', fontsize=10)
              
ax[0, 1].plot(range(1, len(model2_train_acc)+1), model2_train_acc, 
              'c-', label='Training Acc')
ax[0, 1].set_title('LSTM with 784 steps')
ax[0, 1].set_xlabel('Training Iterations', fontsize=9)
ax[0, 1].legend(loc='best', fontsize=10)

ax[1, 0].plot(range(1, len(model3_train_acc)+1), model3_train_acc, 
              'r-', label='Training Acc')
ax[1, 0].set_title('Basic RNN with 28 steps')
ax[1, 0].set_xlabel('Training Iterations', fontsize=9)
ax[1, 0].set_ylabel('Accuracy')
ax[1, 0].legend(loc=4, fontsize=10)

ax[1, 1].plot(range(1, len(model4_train_acc)+1), model4_train_acc, 
              'y-', label='Training Acc')
ax[1, 1].set_title('LSTM with 28 steps')
ax[1, 1].set_xlabel('Training Iterations', fontsize=9)
ax[1, 1].legend(loc=4, fontsize=10)

fig.savefig('./result/convergence.pdf')
fig.savefig('./result/convergence.png', dpi=300)
plt.show()



