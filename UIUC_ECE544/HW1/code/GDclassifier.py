#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "09/13/2016"
"""

import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

class GDclassifier(object):
    """ self-definde Gradient Descent classifier """
    def __init__(self, loss, learning_rate=0.01, iteration=1000, C=None,
                 CV=False, showFreq=100, randomSeed=None):
        self.loss = loss
        self.learning_rate  = learning_rate
        self.iteration = iteration
        self.C = C
        self.CV = CV
        self.showFreq = showFreq
        self.randomSeed = randomSeed
        self.trainError = []
        self.trainAcc = []
        self.cvError = []
        self.cvAcc = []
        self.w = []


    def train(self, X, y, X_cv=None, y_cv=None, verbose=False):
        """ function to train the model
            Note: if you want to train the format g(w'x + b)
                  you need to add 1 to X manually
                  the program will not do this automatically
        """
        np.random.seed(seed=self.randomSeed)

        n, m = X.shape  # n samples, m features
        if self.loss == 'linear':
            w = np.array([np.zeros(m)]).T  # dim: m by 1
        elif self.loss == 'logistic':
            w = np.array([np.random.rand(m)]).T / 1  # dim: m by 1
            # w = np.array([np.zeros(m)]).T  # dim: m by 1
        elif self.loss == 'perceptron':
            w = np.array([np.random.rand(m)]).T  # dim: m by 1
            # w = np.array([np.zeros(m)]).T  # dim: m by 1
        elif self.loss == 'svm':
            w = np.array([np.random.rand(m)]).T / 5  # dim: m by 1
            # w = np.array([np.zeros(m)]).T  # dim: m by 1

        for i in range(1, self.iteration + 1):
            gradient = self.computeGradient(X, y, w)
            w = w - self.learning_rate * gradient / n
            Error, Acc = self.evaluate(X, y, w)
            self.trainError.append(Error)
            self.trainAcc.append(Acc)
            self.w.append(w)
            # evaluate on the cross-validation set
            if self.CV == True:
                tmp_cv_Error, tmp_cv_Acc = self.evaluate(X_cv, y_cv, w)
                self.cvError.append(tmp_cv_Error)
                self.cvAcc.append(tmp_cv_Acc)

            # print current process
            if verbose == True and self.showFreq != 0 and i % self.showFreq == 0:
                print(str(i) + "th Iteration, ", "Error: ", Error, " Accuracy : ", Acc)
                if self.CV == True:
                    print("Cross-Validation: ", "Error: ", tmp_cv_Error, " Accuracy : ", tmp_cv_Acc)
        if verbose == True:
            print("Reach the Maximum Iteration : " + str(i) + "th Iteration")
            bestError, bestAcc, bestW = self.getBest("Accuracy")
            print("Best Training Error: ", bestError, " Highest Training Accuracy : ", bestAcc)
            if self.CV == True:
                best_cv_Error, best_cv_Acc = self.evaluate(X_cv, y_cv, bestW)
                print("Best Development Error: ", best_cv_Error, " Highest Development Accuracy : ", best_cv_Acc)



    def predict(self, X, w):
        """ function to predict X using calculated w """
        if self.loss == 'linear':
            value = X.dot(w)
            prediction = (value >= 0.5).astype(int)
        elif self.loss == 'perceptron':
            value = X.dot(w)
            prediction = (value >= 0).astype(int)
        elif self.loss == 'svm':
            value = X.dot(w)
            prediction = (value >= 0).astype(int)
        elif self.loss == 'logistic':
            value = self.logistic(X, w)
            prediction = (value >= 0.5).astype(int)

        return value, prediction


    def evaluate(self, X, y, w):
        """ function to evaluete the performance using Error and Accuracy """
        value, prediction = self.predict(X, w)
        if self.loss == 'linear' or self.loss == 'logistic':
            Error = np.sum((value - y) ** 2)
        elif self.loss == 'perceptron':
            newY = (y > 0).astype(int) * 2 - 1  # change from (0, 1) to (-1, 1)
            tmp = - value * newY
            Error = np.sum(tmp[tmp > 0])
        elif self.loss == 'svm':
            newY = (y > 0).astype(int) * 2 - 1  # change from (0, 1) to (-1, 1)
            tmp = 1 - value * newY
            h = np.sum(tmp[tmp > 0])
            Error = np.sum(w ** 2) + self.C * h

        Error = Error / len(y)
        Acc = np.sum(prediction == y) / len(y)

        return Error, Acc


    def computeGradient(self, X, y, w):
        """ function to compute the gradient """
        n = len(X)
        if self.loss == 'linear':
            gradient = -2 * np.dot(X.T, (y - X.dot(w)))
        elif self.loss == 'logistic':
            g = self.logistic(X, w)
            gradient = -2 * np.dot(X.T, (y - g) * g * (1 - g))
        elif self.loss == 'perceptron':
            newY = (y > 0).astype(int) * 2 - 1  # change from (0, 1) to (-1, 1)
            index = ((np.dot(X, w) >= 0).astype(int) != y)
            usedX = X[index[:, 0]]
            usedY = newY[index[:, 0]]
            gradient = -np.dot(usedX.T, usedY)
        elif self.loss == 'svm':
            newY = (y > 0).astype(int) * 2 - 1  # change from (0, 1) to (-1, 1)
            index = (np.dot(X, w) * newY < 1)
            usedX = X[index[:, 0]]
            usedY = newY[index[:, 0]]
            gradient = 2 * w - self.C * np.dot(usedX.T, usedY)
            gradient[0] = gradient[0] + 2 * w[0]

        return gradient


    def logistic(self, X, w):
        """ function to calculate the gradient of logistic function """
        g = 1 / (1 + np.exp(-X.dot(w)))
        return g


    def getParams(self):
        """ function to get the parameters """
        return self.trainError, self.trainAcc, self.w


    def getBest(self, category):
        """ function to get the best parameter combination according to highest accuracy """
        if category == 'Accuracy':
            index = np.argmax(self.trainAcc)
        elif category == 'Error':
            index = np.argmin(self.trainError)

        return self.trainError[index], self.trainAcc[index], self.w[index]


    def plot(self, ylog=False, category="Accuracy", figsize=(12, 5)):
        """ function to plot the training and CV error and error rate """
        if self.CV == False:  # no Cross Validation set case
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            plt.suptitle("Training Curve for " + self.loss, fontsize=12)
            ax[0].plot(range(1, len(self.trainError) + 1), self.trainError, 'g-', label='Training Error')
            ax[0].set_xlabel('Iteration')
            ax[0].set_ylabel("Error")
            if ylog == True:
                ax[0].set_yscale('log')
            ax[0].legend()
            ax[0].grid('on')

            if category == "Accuracy":
                ax[1].plot(range(1, len(self.trainAcc) + 1), self.trainAcc, 'r-', label='Training Accuracy')
                ax[1].set_ylabel("Accuracy")
            elif category == "Error Rate":
                ax[1].plot(range(1, len(self.trainAcc) + 1), 1 - np.array(self.trainAcc), 'r-', label='Training Error Rate')
                ax[1].set_ylabel("Error Rate")
            # ax[1].set_ylim((0, 1))
            ax[1].set_xlabel('Iteration')
            ax[1].legend(loc='best')
            ax[1].grid('on')
            plt.show()
        if self.CV == True:  # has Cross Validation set case
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            plt.suptitle("Training Curve for " + self.loss, fontsize=12)
            ax[0].plot(range(1, len(self.trainError) + 1), self.trainError, 'g-', label='Training Error')
            ax[0].plot(range(1, len(self.cvError) + 1), self.cvError, 'r-', label='CV Error')
            ax[0].set_xlabel('Iteration')
            ax[0].set_ylabel("Error")
            if ylog == True:
                ax[0].set_yscale('log')
            ax[0].legend()
            ax[0].grid('on')

            if category == "Accuracy":
                ax[1].plot(range(1, len(self.trainAcc) + 1), self.trainAcc, 'g-', label='Training Accuracy')
                ax[1].plot(range(1, len(self.cvAcc) + 1), self.cvAcc, 'r-', label='CV Accuracy')
                ax[1].set_ylabel("Accuracy")
            elif category == "Error Rate":
                ax[1].plot(range(1, len(self.trainAcc) + 1), 1 - np.array(self.trainAcc), 'g-', label='Training Error Rate')
                ax[1].plot(range(1, len(self.cvAcc) + 1), 1 - np.array(self.cvAcc), 'r-', label='CV Error Rate')
                ax[1].set_ylabel("Error Rate")
            # ax[1].set_ylim((0, 1))
            ax[1].set_xlabel('Iteration')
            ax[1].legend(loc='best')
            ax[1].grid('on')
            plt.show()

        return fig, ax
