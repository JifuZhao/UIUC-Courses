#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "10/28/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import scipy as sp
from scipy.stats import truncnorm
import time


class RBM(object):
    """ self-defined class for Restricted Boltzmann Machine (RBM)"""

    def __init__(self, hidden_nodes, learning_rate, batch_size, n_iter):
        """ initialize the RBM """
        self.n = hidden_nodes
        self.m = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.W = None
        self.b = None
        self.c = None

    def _initialize(self, X):
        """ function to initialize W, b and c """
        generator = truncnorm(a=0, b=1, scale=1)
        self.W = generator.rvs((self.n, self.m))
        self.b = generator.rvs((self.m, 1))
        self.c = generator.rvs((self.n, 1))


    def train(self, X):
        """ function to train the RBM """
        t0 = time.time()
        N_sample, self.m = X.shape
        self._initialize(X)  # initialize W, b, c
        if N_sample % self.batch_size == 0:
            N_batch = N_sample // self.batch_size
        else:
            N_batch = N_sample // self.batch_size + 1

        for iteration in range(1, self.n_iter + 1):
            for i in range(N_batch):
                v0 = X[i*self.batch_size: (i+1)*self.batch_size, :].T
                h_prob, h = self.update_h(v0)
                v1 = self.update_v(h)
                dW, db, dc = self.gradient(v0, h, v1, h_prob)
                self.W += self.learning_rate * dW
                self.b += self.learning_rate * db
                self.c += self.learning_rate * dc

            if self.n_iter > 1:
                print('Finish the ' + str(iteration) + ' th iteration !')

        t = np.round(time.time() - t0, 2)
        print('Reach the ' + str(iteration) + ' th iteration, used time ' + str(t) + 's !')


    def update_h(self, v):
        """ function to sample the hidden nodes """
        tmp = np.dot(self.W, v) + self.c
        prob = self.sigmoid(tmp)
        h = np.random.binomial(1, prob)

        return prob, h


    def update_v(self, h):
        """ function to sample the visible nodes """
        tmp = np.dot(self.W.T, h) + self.b
        prob = self.sigmoid(tmp)
        v = np.random.binomial(1, prob)

        return v


    def gradient(self, v0, h, v1, h_prob_v0):
        """ function to compute the gradient for W, b, c """
        tmp_h_v1 = np.dot(self.W, v1) + self.c
        h_prob_v1 = self.sigmoid(tmp_h_v1)

        dW = np.dot(h_prob_v0, v0.T) - np.dot(h_prob_v1, v1.T)
        db = np.mean(v0 - v1, axis=1)
        dc = np.mean(h_prob_v0 - h_prob_v1, axis=1)
        dW /= self.batch_size

        return dW, db, dc


    def sigmoid(self, t):
        """ function to compute the sigmoid """
        return 1 / (1 + np.exp(-t))


    def get_params(self):
        """ funtion to get the trained parameters """
        return self.W, self.b, self.c
