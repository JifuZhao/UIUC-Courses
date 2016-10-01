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
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time

class NeuralNetwork(object):
    """ self-defined Neural Network classifier """

    def __init__(self, netSize, loss, maxIter=500, batchSize=100,
                 learningRate=0.1, CV=False):
        self.netSize = netSize
        self.loss = loss
        self.originalLearningRate = learningRate
        self.learningRate = None
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.CV = CV
        self.trainAcc = []
        self.cvAcc = []
        self.wList = []
        self.w = []
        self.layer = 1 + len(netSize)
        self.time = []


    def train(self, X, y, X_cv=None, y_cv=None, showFreq=100):
        """ function to train the neural network """

        # randomly initialize the weight matrix w
        numNode = [len(X[0])] + list(self.netSize)
        for i in range(self.layer - 1):
            randomW = (np.random.random((numNode[i + 1], numNode[i] + 1)) - 0.5)
            self.w.append(randomW)

        t0 = time.time()
        # begin training process
        for iterate in range(1, self.maxIter + 1):
            self.learningRate = min(self.originalLearningRate,
                                    self.originalLearningRate / (iterate / 50))

            X, y = shuffle(X, y)
            batchesX, batchesY = self.getBatches(X, y)
            for batchFeature, BatchLabel in zip(batchesX, batchesY):
                tStart = time.time()
                self.w = self.updateW(batchFeature, BatchLabel, self.w)
                self.time.append(time.time() - tStart)

            self.wList.append(self.w)
            self.trainAcc.append(self.evaluate(X, y, self.w))
            if self.CV == True:
                self.cvAcc.append(self.evaluate(X_cv, y_cv, self.w))

            if (iterate % showFreq == 0):
                t = np.round(time.time() - t0, 2)
                print(iterate, "th Iteration is done, used time\t", t, 's')

        t = np.round(time.time() - t0, 2)
        print("Reach the maximum iteration\t", t, 's')
        t = np.round(np.mean(self.time), 5)
        print('Used time for one iteration (single batch): \t', t, 's')


    def updateW(self, X, y, w):
        """ function to update the calculated w """
        n = len(X)
        g = [self.addBias(X)]  # g means x, h(w'x)
        z = [X]  # z means x, w'x, w'g(w'x)

        for i in range(self.layer - 2):
            forwardW = w[i]
            feature = g[i]
            In = np.dot(feature, forwardW.T)
            z.append(In)
            Out = self.nonLinearity(In)
            g.append(self.addBias(Out))

        forwardW = w[self.layer - 2]
        feature = g[self.layer - 2]
        In = np.dot(feature, forwardW.T)
        z.append(In)
        Out = self.softmax(In)
        g.append(Out)

        gradient = self.computeGradient(g, z, w, y)
        newW = []
        for i in range(self.layer - 1):
            newW.append(w[i] - self.learningRate * gradient[i] / n)

        return newW


    def computeGradient(self, g, z, w, y):
        """ function to compute the gradient """
        gradient = []
        label = np.argmax(y, axis=1)
        b = g[-1]
        b[range(len(b)), label] -= 1

        for i in reversed(range(self.layer - 1)):
            gradient.append(np.dot(b.T, g[i]))
            b = (np.dot(b, w[i])[:, 1:]) * self.backGradient(z[i])

        return gradient[::-1]


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


    def predict(self, X, w):
        """ function to predict X based on trained model """
        feature = X

        for i in range(self.layer - 2):
            weight = w[i]
            feature = self.addBias(feature)
            feature = self.nonLinearity(np.dot(feature, weight.T))

        weight = w[self.layer - 2]
        feature = self.addBias(feature)
        feature = self.softmax(np.dot(feature, weight.T))

        prediction = np.argmax(feature, axis=1)
        return prediction


    def evaluate(self, X, y, w):
        """ function to evaluate the performance of trained model """
        prediction = self.predict(X, w)
        label = np.argmax(y, axis=1)
        acc = np.sum(prediction == label) / len(prediction)

        return acc


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


    def softmax(self, z):
        """ function to calculate the softmax function """
        ans = np.exp(z)
        ans = ans.T / np.sum(ans, axis=1)
        ans = ans.T
        return ans


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
        if self.loss == 'relu':
            data = np.append(np.array([np.ones(n)]).T, data, axis=1)
        else:
            data = np.append(np.array([np.ones(n)]).T, data, axis=1)
        return data


    def getParams(self):
        """ function to get the parameters """
        if self.CV == True:
            return self.trainAcc, self.cvAcc, self.wList
        else:
            return self.trainAcc, self.wList


    def getBest(self):
        """ function to get the best parameter combinations """
        index = np.argmax(self.trainAcc)

        return self.trainAcc[index], self.wList[index]
