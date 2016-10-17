#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "10/16/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

class EM(object):
    """ self-defined calss for EM algorithm """

    def __init__(self, m, threshold=0.01, maxIter=500):
        """ initialize the EM algorithm """
        self.m = m
        self.threshold = threshold
        self.maxIter = maxIter
        self.w = None
        self.gamma = None
        self.mu = None
        self.sigma = None
        self.gaussianProb = None
        self.E = []


    def train(self, x):
        """ function to perform EM algorithm on X """
        # initialize the mean and covariance matrix
        self.initialize(x)

        # iterate through E and M steps
        for i in range(1, self.maxIter+1):
            self.estep(x)
            self.mstep(x)
            if abs(self.E[-1] - self.E[-2]) < self.threshold:
                print('Reach the threshold at', i, 'th iteration !')

        print('Reach the maximum iteration !')


    def initialize(self, x):
        """ function to initialize the parameters """
        n, dim = x.shape  # find the dimensions
        self.w = np.ones(self.m) * (1 / self.m)
        self.gamma = np.zeros((n, self.m))
        self.mu = np.zeros((self.m, dim))
        self.sigma = np.zeros((self.m, dim, dim))
        self.gaussianProb = np.zeros((n, self.m))

        maximum = np.amax(x)
        minimum = np.amin(x)
        diag = np.cov(x.T).diagonal()
        for k in range(self.m):
            self.mu[k, :] = np.random.uniform(minimum, maximum, dim)
            diaginal = np.random.uniform()
            self.sigma[k, :, :] = np.diag(diag)
            self.gaussianProb[:, k] = self.gaussian(x, self.mu[k, :], self.sigma[k, :, :])

        weightedSum = np.sum(self.w * self.gaussianProb, axis=1)
        for k in range(self.m):
            self.gamma[:, k] = self.w[k] * self.gaussianProb[:, k] / weightedSum

        # calculate the expectation of log-likelihood
        self.E.append(self.calculateE())


    def estep(self, x):
        """ function to conduct E-Step for EM algorithm """
        for k in range(self.m):
            self.gaussianProb[:, k] = self.gaussian(x, self.mu[k, :], self.sigma[k, :, :])

        weightedSum = np.sum(self.w * self.gaussianProb, axis=1)
        for k in range(self.m):
            self.gamma[:, k] = self.w[k] * self.gaussianProb[:, k] / weightedSum


    def mstep(self, x):
        """ function to conduct M-Step for EM algorithm """
        sumGamma = np.sum(self.gamma, axis=0)
        self.w = np.mean(self.gamma, axis=0)
        for k in range(self.m):
            self.mu[k, :] = np.sum(x.T * self.gamma[:, k], axis=1) / sumGamma[k]
            diff = x - self.mu[k, :]
            weightedDiff = diff.T * self.gamma[:, k]
            self.sigma[k, :, :] = np.dot(weightedDiff, diff) / sumGamma[k]

        # calculate the expectation of log-likelihood
        self.E.append(self.calculateE())


    def gammaprob(self, x):
        """ function to calculate the gamma probability """
        pass


    def gaussian(self, x, mu, sigma):
        """ function to calculate the multivariate gaussian probability """
        pdf = multivariate_normal(mu, sigma).pdf(x)

        return pdf


    def calculateE(self):
        """ function to calculate the expection of log likelihood """
        log = np.log(self.w * self.gaussianProb)
        E = np.sum(log * self.gamma)

        return E


    def predict(self, x):
        """ function to predict the classes using calculated parameters """
        pass


    def get_params(self):
        """ function to return the calculated parameters """
        return self.gamma, self.mu, self.sigma, self.E
