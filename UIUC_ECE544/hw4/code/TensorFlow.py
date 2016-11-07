#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/04/2016"

"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
import tflearn

from tf_RBM import RBM
from tf_RBM import showConfusionMatrix
from tf_RBM import oneHotEncoder


# load the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_img = mnist.train.images
train_label = mnist.train.labels

test_img = mnist.test.images
test_label = mnist.test.labels

# get the label in single number case
train_true_label = np.argmax(train_label, axis=1)
test_true_label = np.argmax(test_label, axis=1)


# # Classification using the raw-pixel data
# Classification using tflearn
with tf.Graph().as_default():
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.6)

    # build the one-layer fully connected neural network
    net = tflearn.input_data(shape=[None, 784])
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    # fit the model
    model = tflearn.DNN(net)
    model.fit(train_img, train_label)

    # predict on the training and test dataset
    train_prediction = np.array(model.predict(train_img))
    test_prediction = np.array(model.predict(test_img))

# get the label
train_pred_label = np.argmax(train_prediction, axis=1)
test_pred_label = np.argmax(test_prediction, axis=1)

# calculate the training and testing accuracy
train_acc = np.sum(train_pred_label == train_true_label) / len(train_true_label)
test_acc = np.sum(test_pred_label == test_true_label) / len(test_true_label)
print('Training Accuract:', str(np.round(train_acc, 5)))
print('Testing Accuract:', str(np.round(test_acc, 5)))

# calculate the confusion matrix
trainMatrix = np.dot(train_label.T, oneHotEncoder(np.array([train_pred_label]).T, 10))
testMatrix = np.dot(test_label.T, oneHotEncoder(np.array([test_pred_label]).T, 10))

# plot the training and test confusion matrix
label=[str(i) for i in range(10)]
title1 = 'Training Confusion Matrix'
fig1 = showConfusionMatrix(trainMatrix.astype(int), title=title1, label=label, fontsize=10)
fig1.savefig('./result/train_raw.pdf')
plt.show()

title2 = 'Testing Confusion Matrix'
fig2 = showConfusionMatrix(testMatrix.astype(int), title=title2, label=label, fontsize=10)
fig2.savefig('./result/test_raw.pdf')
plt.show()


# # Classification using single layer RBM
# build the RBM model
model = RBM(visible_nodes=784, hidden_nodes=200, learning_rate=0.001, batch_size=10, n_iter=3)
W, b, c = model.train(train_img)

# calculate the output of RBM on training and testing dataset
train_rbm = 1 / (1 + np.exp(-np.dot(train_img, W.T) + c))
test_rbm = 1 / (1 + np.exp(-np.dot(test_img, W.T) + c))

# Classification using tflearn
with tf.Graph().as_default():
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.6)

    # build the one-layer fully connected neural network
    net = tflearn.input_data(shape=[None, 200])
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    # fit the model
    model = tflearn.DNN(net)
    model.fit(train_rbm, train_label)

    # predict on the training and test dataset
    train_prediction = np.array(model.predict(train_rbm))
    test_prediction = np.array(model.predict(test_rbm))

# get the label
train_pred_label = np.argmax(train_prediction, axis=1)
test_pred_label = np.argmax(test_prediction, axis=1)

# calculate the training and testing accuracy
train_acc = np.sum(train_pred_label == train_true_label) / len(train_true_label)
test_acc = np.sum(test_pred_label == test_true_label) / len(test_true_label)
print('Training Accuract:', str(np.round(train_acc, 5)))
print('Testing Accuract:', str(np.round(test_acc, 5)))

# calculate the confusion matrix
trainMatrix = np.dot(train_label.T, oneHotEncoder(np.array([train_pred_label]).T, 10))
testMatrix = np.dot(test_label.T, oneHotEncoder(np.array([test_pred_label]).T, 10))

# plot the training and test confusion matrix
label=[str(i) for i in range(10)]
title1 = 'Training Confusion Matrix'
fig1 = showConfusionMatrix(trainMatrix.astype(int), title=title1, label=label, fontsize=10)
fig1.savefig('./result/train_rbm.pdf')
plt.show()

title2 = 'Testing Confusion Matrix'
fig2 = showConfusionMatrix(testMatrix.astype(int), title=title2, label=label, fontsize=10)
fig2.savefig('./result/test_rbm.pdf')
plt.show()


# # Classification using PCA
# build the PCA model
pca = PCA(n_components=200, whiten=False)
pca.fit(train_img)

train_pca = pca.transform(train_img)
test_pca = pca.transform(test_img)

# Classification using tflearn
with tf.Graph().as_default():
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.6)

    # build the one-layer fully connected neural network
    net = tflearn.input_data(shape=[None, 200])
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    # fit the model
    model = tflearn.DNN(net)
    model.fit(train_pca, train_label)

    # predict on the training and test dataset
    train_prediction = np.array(model.predict(train_pca))
    test_prediction = np.array(model.predict(test_pca))

# get the label
train_pred_label = np.argmax(train_prediction, axis=1)
test_pred_label = np.argmax(test_prediction, axis=1)

# calculate the training and testing accuracy
train_acc = np.sum(train_pred_label == train_true_label) / len(train_true_label)
test_acc = np.sum(test_pred_label == test_true_label) / len(test_true_label)
print('Training Accuract:', str(np.round(train_acc, 5)))
print('Testing Accuract:', str(np.round(test_acc, 5)))

# calculate the confusion matrix
trainMatrix = np.dot(train_label.T, oneHotEncoder(np.array([train_pred_label]).T, 10))
testMatrix = np.dot(test_label.T, oneHotEncoder(np.array([test_pred_label]).T, 10))

# plot the training and test confusion matrix
label=[str(i) for i in range(10)]
title1 = 'Training Confusion Matrix'
fig1 = showConfusionMatrix(trainMatrix.astype(int), title=title1, label=label, fontsize=10)
fig1.savefig('./result/train_pca.pdf')
plt.show()

title2 = 'Testing Confusion Matrix'
fig2 = showConfusionMatrix(testMatrix.astype(int), title=title2, label=label, fontsize=10)
fig2.savefig('./result/test_pca.pdf')
plt.show()


# # Classification using stacking RBM
# build the first RBM model
model = RBM(visible_nodes=784, hidden_nodes=500, learning_rate=0.001, batch_size=10, n_iter=3)
W1, b1, c1 = model.train(train_img)

# calculate the output of the first RBM on training and testing dataset
train_rbm1 = 1 / (1 + np.exp(-np.dot(train_img, W1.T) + c1))
test_rbm1 = 1 / (1 + np.exp(-np.dot(test_img, W1.T) + c1))

# build the second RBM model
model = RBM(visible_nodes=500, hidden_nodes=200, learning_rate=0.01, batch_size=10, n_iter=3)
W2, b2, c2 = model.train(train_rbm1)

# calculate the output of RBM on training and testing dataset
train_rbm2 = 1 / (1 + np.exp(-np.dot(train_rbm1, W2.T) + c2))
test_rbm2 = 1 / (1 + np.exp(-np.dot(test_rbm1, W2.T) + c2))

# Classification using tflearn
with tf.Graph().as_default():
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.6)

    # build the one-layer fully connected neural network
    net = tflearn.input_data(shape=[None, 200])
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    # fit the model
    model = tflearn.DNN(net)
    model.fit(train_rbm2, train_label)

    # predict on the training and test dataset
    train_prediction = np.array(model.predict(train_rbm2))
    test_prediction = np.array(model.predict(test_rbm2))

# get the label
train_pred_label = np.argmax(train_prediction, axis=1)
test_pred_label = np.argmax(test_prediction, axis=1)

# calculate the training and testing accuracy
train_acc = np.sum(train_pred_label == train_true_label) / len(train_true_label)
test_acc = np.sum(test_pred_label == test_true_label) / len(test_true_label)
print('Training Accuract:', str(np.round(train_acc, 5)))
print('Testing Accuract:', str(np.round(test_acc, 5)))

# calculate the confusion matrix
trainMatrix = np.dot(train_label.T, oneHotEncoder(np.array([train_pred_label]).T, 10))
testMatrix = np.dot(test_label.T, oneHotEncoder(np.array([test_pred_label]).T, 10))

# plot the training and test confusion matrix
label=[str(i) for i in range(10)]
title1 = 'Training Confusion Matrix'
fig1 = showConfusionMatrix(trainMatrix.astype(int), title=title1, label=label, fontsize=10)
fig1.savefig('./result/train_stacking.pdf')
plt.show()

title2 = 'Testing Confusion Matrix'
fig2 = showConfusionMatrix(testMatrix.astype(int), title=title2, label=label, fontsize=10)
fig2.savefig('./result/test_stacking.pdf')
plt.show()




