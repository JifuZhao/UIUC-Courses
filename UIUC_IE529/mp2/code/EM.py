#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "12/08/2016"
"""

import numpy as np
import time


class EM(object):
    """ self-defined calss for EM algorithm """

    def __init__(self, m, threshold=0.01, random_state=None, maxIter=500):
        """ initialize the EM algorithm """
        self.m = m
        self.threshold = threshold
        self.random_state = random_state
        self.maxIter = maxIter
        self.w = None
        self.gamma = None
        self.mu = None
        self.sigma = None
        self.gaussianProb = None
        self.logLikelihood = None

    def train(self, x, verbose=False):
        """ function to perform EM algorithm on X """
        t0 = time.time()
        np.random.seed(self.random_state)

        # initialize the mean and covariance matrix
        self.initialize(x)

        # iterate through E and M steps
        for i in range(1, self.maxIter + 1):
            self.estep(x)
            self.mstep(x)
            if abs(self.logLikelihood[-1] - self.logLikelihood[-2]) \
               / abs(self.logLikelihood[-2]) < self.threshold:
                t = np.round(time.time() - t0, 4)
                print('Reach threshold at', i, 'th iters in ' + str(t) + 's')
                return
        t = np.round(time.time() - t0, 4)
        print('Stopped, reach the maximum iteration ' + str(t) + 's')

    def initialize(self, x):
        """ function to initialize the parameters """
        n, dim = x.shape  # find the dimensions
        self.w = np.ones(self.m) * (1 / self.m)
        self.gamma = np.zeros((n, self.m))
        self.gaussianProb = np.zeros((n, self.m))
        self.mu = [None] * self.m
        self.sigma = [None] * self.m
        self.logLikelihood = []

        cov = np.cov(x.T)
        mean = np.mean(x, axis=0)
        for k in range(self.m):
            self.mu[k] = mean + np.random.uniform(-0.5, 0.5, dim)
            self.sigma[k] = cov

        # update gamma
        self.gamma = self.gammaprob(x, self.w, self.mu, self.sigma)

        # calculate the expectation of log-likelihood
        self.logLikelihood.append(self.likelihood())

    def estep(self, x):
        """ function to conduct E-Step for EM algorithm """
        self.gamma = self.gammaprob(x, self.w, self.mu, self.sigma)

    def mstep(self, x):
        """ function to conduct M-Step for EM algorithm """
        n, dim = x.shape
        sumGamma = np.sum(self.gamma, axis=0)
        self.w = sumGamma / n

        for k in range(self.m):
            self.mu[k] = np.sum(x.T * self.gamma[:, k], axis=1) / sumGamma[k]
            diff = x - self.mu[k]
            weightedDiff = diff.T * self.gamma[:, k]
            self.sigma[k] = np.dot(weightedDiff, diff) / sumGamma[k]
            if np.linalg.matrix_rank(self.sigma[k]) != 3:
                randv = np.random.random(dim) / 10000
                self.sigma[k] = self.sigma[k] + np.diag(randv)

        # calculate the expectation of log-likelihood
        self.logLikelihood.append(self.likelihood())

    def gammaprob(self, x, w, mu, sigma):
        """ function to calculate the gamma probability """
        for k in range(self.m):
            self.gaussianProb[:, k] = self.gaussian(x, mu[k], sigma[k])

        weightedSum = np.sum(w * self.gaussianProb, axis=1)
        gamma = ((w * self.gaussianProb).T / weightedSum).T

        return gamma

    def gaussian(self, x, mu, sigma):
        """ function to calculate the multivariate gaussian probability """
        inversion = np.linalg.inv(sigma)
        part1 = (-0.5 * np.sum(np.dot(x - mu, inversion) * (x - mu), axis=1))
        part2 = 1 / ((2 * np.pi) ** (len(mu) / 2) *
                     (np.linalg.det(sigma) ** 0.5))

        pdf = part2 * np.exp(part1)

        return pdf

    def likelihood(self):
        """ function to calculate the log likelihood """
        log = np.log(np.sum(self.w * self.gaussianProb, axis=1))
        logLikelihood = np.sum(log)

        return logLikelihood

    def get_label(self):
        """ function to predict the classes using calculated parameters """
        label = np.argmax(self.w * self.gaussianProb, axis=1)

        return label


def main():
    """ implement the EM algorithm on real-world data """
    print('test case !')


if __name__ == '__main__':
    main()
