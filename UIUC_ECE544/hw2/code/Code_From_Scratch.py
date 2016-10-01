#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "09/27/2016"
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from helper import readFile
from helper import oneHotEncoder
from helper import showConfusionMarix
from NeuralNetwork import NeuralNetwork


# ### Load data and One-Hot-Encode
# read the data
trainFeature, trainLabel = readFile('./data/train/lab/hw2train_labels.txt', './data/')
devFeature, devLabel = readFile('./data/dev/lab/hw2dev_labels.txt', './data/')
evalFeature, evalLabel = readFile('./data/eval/lab/hw2eval_labels.txt', './data/')

# One-Hot-Encode for labels
trainLabel = oneHotEncoder(trainLabel, 9)
devLabel = oneHotEncoder(devLabel, 9)
evalLabel = oneHotEncoder(evalLabel, 9)

# Universial Hyper-parameters
batchSize = 50
show = False

print('*' * 60)
print('Batch Size is:\t', batchSize, '\n')


# ### Relu non-linearity

print('ReLU Function')
hiddenNodeList = [10, 20, 30, 40, 50]
learningRateList = [0.1, 0.09, 0.08, 0.08, 0.06]

for i in range(5):
    hidden = hiddenNodeList[i]
    learningRate = learningRateList[i]
    print('*' * 60)
    print('*' * 20, ' Hidden node is', hidden, '*' * 20)
    print('*' * 60)
    
    # create the Neural Network classifier
    nn = NeuralNetwork(netSize=(hidden, hidden, 9), loss='relu', maxIter=500, 
                       batchSize=batchSize, learningRate=learningRate, CV=True)

    # train the model
    test = nn.train(trainFeature, trainLabel, devFeature, devLabel, showFreq=1000)
    # get the accuracy
    trainAcc, cvAcc, w = nn.getParams()
    # get the accuracy information
    index = np.argmax(cvAcc[50:]) + 50
    wBest = w[index]
    # get the evaluation accuracy
    testAcc = nn.evaluate(evalFeature, evalLabel, wBest)
    
    print('At', index + 1, 'th iteration, reach the maximum development accuracy')
    print('Training Accuracy:\t', trainAcc[index])
    print('Development Accuracy:\t', cvAcc[index])
    print('Evaluation Accuracy:\t', testAcc)
    
    if show == True:
        # plot the training accuracy
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trainAcc, 'g', label='Training Accuracy')
        ax.plot(cvAcc, 'r', label='Development Accuracy')
        ax.set_title('Convergence curve with ReLU and ' + str(hidden) + ' hidden nodes')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.legend(loc=4, fontsize=10)
        ax.grid('on')
        plt.show()
        
print('\n')


# ### Sigmoid non-linearity

print('Sigmoid Function')
hiddenNodeList = [10, 20, 30, 40, 50]
learningRateList = [0.3, 0.3, 0.3, 0.3, 0.3]

for i in range(5):
    hidden = hiddenNodeList[i]
    learningRate = learningRateList[i]
    print('*' * 60)
    print('*' * 20, ' Hidden node is', hidden, '*' * 20)
    print('*' * 60)
    
    # create the Neural Network classifier
    nn = NeuralNetwork(netSize=(hidden, hidden, 9), loss='sigmoid', maxIter=500, 
                       batchSize=batchSize, learningRate=learningRate, CV=True)

    # train the model
    test = nn.train(trainFeature, trainLabel, devFeature, devLabel, showFreq=1000)
    # get the accuracy
    trainAcc, cvAcc, w = nn.getParams()
    # get the accuracy information
    index = np.argmax(cvAcc[50:]) + 50
    wBest = w[index]
    # get the evaluation accuracy
    testAcc = nn.evaluate(evalFeature, evalLabel, wBest)
    
    print('At', index + 1, 'th iteration, reach the maximum development accuracy')
    print('Training Accuracy:\t', trainAcc[index])
    print('Development Accuracy:\t', cvAcc[index])
    print('Evaluation Accuracy:\t', testAcc)
    
    if show == True:
        # plot the training accuracy
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trainAcc, 'g', label='Training Accuracy')
        ax.plot(cvAcc, 'r', label='Development Accuracy')
        ax.set_title('Convergence curve with Sigmoid and ' + str(hidden) + ' hidden nodes')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.legend(loc=4, fontsize=10)
        ax.grid('on')
        plt.show()
        
print('\n')


# ### Tanh non-linearity

print('Tahh Function')
hiddenNodeList = [10, 20, 30, 40, 50]
learningRateList = [0.3, 0.3, 0.3, 0.3, 0.3]

for i in range(5):
    hidden = hiddenNodeList[i]
    learningRate = learningRateList[i]
    print('*' * 60)
    print('*' * 20, ' Hidden node is', hidden, '*' * 20)
    print('*' * 60)

    # create the Neural Network classifier
    nn = NeuralNetwork(netSize=(hidden, hidden, 9), loss='tanh', maxIter=500, 
                       batchSize=batchSize, learningRate=learningRate, CV=True)

    # train the model
    test = nn.train(trainFeature, trainLabel, devFeature, devLabel, showFreq=1000)
    # get the accuracy
    trainAcc, cvAcc, w = nn.getParams()
    # get the accuracy information
    index = np.argmax(cvAcc[50:]) + 50
    wBest = w[index]
    # get the evaluation accuracy
    testAcc = nn.evaluate(evalFeature, evalLabel, wBest)
    
    print('At', index + 1, 'th iteration, reach the maximum development accuracy')
    print('Training Accuracy:\t', trainAcc[index])
    print('Development Accuracy:\t', cvAcc[index])
    print('Evaluation Accuracy:\t', testAcc)
    
    if show == True:
        # plot the training accuracy
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trainAcc, 'g', label='Training Accuracy')
        ax.plot(cvAcc, 'r', label='Development Accuracy')
        ax.set_title('Convergence curve with Tanh and ' + str(hidden) + ' hidden nodes')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.legend(loc=4, fontsize=10)
        ax.grid('on')
        plt.show()
        
print('\n')


# ### Best Model
# 
# * Sigmoid activation function
# * 50 hidden nodes
# * 0.3 learning rate
# * 50 batch size

# create the Neural Network classifier
nn = NeuralNetwork(netSize=(50, 50, 9), loss='sigmoid', maxIter=500, 
                   batchSize=50, learningRate=0.3, CV=True)

# train the model
test = nn.train(trainFeature, trainLabel, devFeature, devLabel, showFreq=1000)
# get the accuracy
trainAcc, cvAcc, w = nn.getParams()
# get the accuracy information
index = np.argmax(cvAcc[50:]) + 50
wBest = w[index]
# get the evaluation accuracy
testAcc = nn.evaluate(evalFeature, evalLabel, wBest)

print('At', index + 1, 'th iteration, reach the maximum development accuracy')
print('Training Accuracy:\t', trainAcc[index])
print('Development Accuracy:\t', cvAcc[index])
print('Evaluation Accuracy:\t', testAcc)

if show == True:
    # plot the training accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trainAcc, 'g', label='Training Accuracy')
    ax.plot(cvAcc, 'r', label='Development Accuracy')
    ax.set_title('Convergence curve with Sigmoid and ' + str(hidden) + ' hidden nodes')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(loc=4, fontsize=10)
    ax.grid('on')
    plt.show()
        
print('\n')


# get the train label and predction
trainPredict = oneHotEncoder(np.array([nn.predict(trainFeature, wBest)]).T, 9)
trainMatrix = np.dot(trainLabel.T, trainPredict)

# get the test label and predction
testPredict = oneHotEncoder(np.array([nn.predict(evalFeature, wBest)]).T, 9)
testMatrix = np.dot(evalLabel.T, testPredict)

# plot the confusion matrix
fig1 = showConfusionMarix(trainMatrix, title='Training Set Confusion Matrix', 
                          label=[str(i) for i in range(1, 10)])
fig1.savefig('./result/trainMatrix.pdf')

fig2 = showConfusionMarix(testMatrix, title='Test Set Confusion Matrix', 
                          label=[str(i) for i in range(1, 10)])
fig2.savefig('./result/testMatrix.pdf')




