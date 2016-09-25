#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "09/21/2016"
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    """ self-defined Neural Network classifier """
    def __init__(self, netSize, loss, maxIter=500, batchSize=100,
                 learningRate=0.1, CV=False):
        self.netSize = netSize
        self.loss = loss
        self.learningRate = learningRate
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.CV = CV
        self.trainAcc = []
        self.cvAcc = []
        self.w = []
        self.layer = 1 + len(netSize)


    def train(self, X, y, X_cv=None, y_cv=None):
        """ function to train the neural network """

        # randomly initialize the weight matrix w
        numNode = [len(X[0])] + list(self.netSize)
        for i in range(self.layer - 1):
            randomW = (np.random.random((numNode[i + 1], numNode[i] + 1)) - 0.5) / 2
            self.w.append(randomW)

        # begin training process
        for iterate in range(self.maxIter):
            X, y = shuffle(X, y)
            batchesX, batchesY = self.getBatches(X, y)
            for batchFeature, BatchLabel in zip(batchesX, batchesY):
                self.w = self.updateW(batchFeature, BatchLabel, self.w)

        print("Reach the maximum iteration, training is done ! \n")


    def updateW(self, X, y, w):
        """ function to update the calculated w """
        n = len(X)
        g = [self.addBias(X)]  # g means x, h(w'x)
        z = [self.addBias(X)]  # z means x, w'x, w'g(w'x)
        for i in range(self.layer - 1):
            forwardW = w[i]
            feature = g[i]
            In = np.dot(feature, forwardW.T)
            z.append(In)
            Out = self.nonLinearity(In)
            if i != self.layer - 2:
                g.append(self.addBias(Out))
            else:
                g.append(Out)

        gradient = self.computeGradient(g, z, w, y)

        newW = []
        for i in range(self.layer):
            newW.append(w[i] - self.learningRate * gradient[i] / n)

        return newW


    def computeGradient(self, g, z, w, y):
        """ function to compute the gradient """
        gradient = []
        b = (g[-1] - y) * self.backGradient(z[-1])
        for i in reversed(range(self.layer - 1)):
            gradient.append(np.dot(b.T, g[i]))
            b = np.dot(w[i].T, b) * self.backGradient(z[i])

        return reversed(gradient)


    def nonLinearity(self, z):
        """ function to perform non-linearity fitting """
        if self.loss == 'sigmoid':
            return self.sigmoid(z)
        elif self.loss == 'tanh':
            return self.tanh(z)
        elif self.loss == 'relu':
            return self.relu(z)


    def backGradient(self, z):
        """ function to calculate the back error """
        if self.loss == 'sigmoid':
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.loss == 'tanh':
            return 1 - self.tanh(z) ** 2
        elif self.loss == 'relu':
            return (z > 0).astype(float)


    def predict(self, X):
        """ function to predict X based on trained model """
        pass


    def evaluate(self, X, y):
        """ function to evaluate the performance of trained model """
        pass


    def sigmoid(self, z):
        """ function to calculate the sigmoid function """
        g = 1 / (1 + np.exp(-z))
        return g


    def tanh(self, z):
        """ function to calculate the tanh function """
        g = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return g


    def relu(self, z):
        """ function to calculate the tanh function """
        g = np.maximum(z, 0)
        return g


    def getBatches(self, X, y):
        """ function to get the batches """
        left = len(X) % self.batchSize
        data = np.concatenate((X, X[:left, :]), axis=0)
        label = np.concatenate((y, y[:left, :]), axis=0)
        batchX = []
        batchY = []
        n = len(data) // self.batchSize
        for i in range(n):
            batchX.append(data[i*self.batchSize: (i + 1) * self.batchSize, :])
            batchY.append(label[i*self.batchSize: (i + 1) * self.batchSize, :])

        return batchX, batchY


    def addBias(self, data):
        """ function to add bias item to the data as the first column """
        n = len(data)
        data = np.append(np.array([np.ones(n)]).T, data, axis=1)
        return data


    def getParams(self):
        """ function to get the parameters """
        return self.trainError, self.trainAcc, self.w


    def getBest(self, category):
        """ function to get the best parameter combinations """
        if category == 'Accuracy':
            index = np.argmax(self.trainAcc)
        elif category == 'Error':
            index = np.argmin(self.trainError)

        return self.trainError[index], self.trainAcc[index], self.w[index]
