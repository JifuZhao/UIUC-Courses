#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/20/2016"
"""
# ------------------------------------------------------------------
# Note: in class kernelPCA(), the kernel matrix is calculated
#       using sklearn built-in function: pairwise_kernels
#       the reason is to speed-up the calculation since our
#       dataset is too big
#-------------------------------------------------------------------

import warnings
warnings.simplefilter('ignore')

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import pairwise_kernels

class kernelPCA(object):
    """ Class to perform kernel PCA on given input X """
    def __init__(self, kernel='polynomial', degree=None, gamma=None, njob=4):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.njob = njob  # njobs for pair_wise kernels
        self.n = None  # length of the input data X
        self.K = None  # kernel matrix K
        self.Kc = None  # centralized kernel matrix K'
        self.X = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.variance = None
    # end __init__()

    def fit(self, X):
        """ function to conduct kernel PCA """
        self.X = X
        params = {'gamma': self.gamma, 'degree': self.degree}
        self.K = pairwise_kernels(X, metric=self.kernel, n_jobs=self.njob, **params)
        self.n = len(X)
        one = np.ones((self.n, self.n))  # helper matrix will all values equal 1

        # centralize the kernel matrix K
        self.Kc = self.K - one.dot(self.K) / self.n - self.K.dot(one) / self.n \
                         + one.dot(self.K).dot(one) / (self.n ** 2)

        # calculate the eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = eigh(self.Kc)
        # re-organize the sequence
        self.eigenvalues = self.eigenvalues[::-1]
        self.eigenvectors = self.eigenvectors[:, ::-1]

        # normalize the eigenvectors
        self.eigenvectors = self.eigenvectors / np.sqrt(self.eigenvalues)
    # end fit()

    def transform(self, Y, same=False):
        """ function to project X onto the first n principal components """
        if same == True:
            result = self.Kc.dot(self.eigenvectors)
        else:
            params = {'gamma': self.gamma, 'degree': self.degree}
            Ktest = pairwise_kernels(Y, self.X, metric=self.kernel, n_jobs=self.njob, **params)
            m = len(Y)
            one_m = np.ones((m, self.n))
            one_n = np.ones((self.n, self.n))
            Ktest_c = Ktest - one_m.dot(self.K) / self.n - Ktest.dot(one_n) / self.n \
                            + one_m.dot(self.K).dot(one_n) / (self.n ** 2)
            result = Ktest_c.dot(self.eigenvectors)

        return result
    # end transform

    def get_result(self, n_components=None):
        """ function to get the first n principal components and n variance """
        if n_components is None:
            print('Keep the all all principal components')
            return self.eigenvectors, self.eigenvalues
        else:
            print('Keep the first ' + str(n_components) + ' principal components')
            return self.eigenvectors[:, :n_components], self.eigenvalues[:n_components]
    # end get_result()
