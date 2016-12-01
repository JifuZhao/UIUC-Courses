#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/20/2016"
"""

import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

class PCA(object):
    """ Class to perform PCA on given input X """
    def __init__(self, n=None):
        self.n = n
        self.eigenvector = None
        self.variance = None
        self.mean = None
    # end __init__()

    def fit(self, X):
        """ function to conduct PCA """
        centeredX = self.center(X)
        covariance = np.dot(centeredX.T, centeredX) / len(X)
        U, S, V = np.linalg.svd(covariance)
        self.eigenvector = U
        self.variance = S
    # end fit()

    def center(self, X):
        """ function to center the input X through substracting the mean value """
        self.mean = np.mean(X, axis=0)
        centeredX = X - self.mean

        return centeredX
    # end center()

    def transform(self, X):
        """ function to project X onto the first n principal components """
        X = X - self.mean
        if self.n is None:
            result = np.dot(X, self.eigenvector)
        else:
            result = np.dot(X, self.eigenvector[:, :self.n])

        return result
    #end transform()

    def get_result(self):
        """ function to get the first n principal components and n variance """
        if self.n is None:
            print('Keep the all ' + str(len(self.variance)) + ' principal components')
            return self.eigenvector, self.variance, self.mean
        else:
            print('Keep the first ' + str(self.n) + ' principal components')
            return self.eigenvector[:, :self.n], self.variance[:self.n], self.mean
    # end get_result()
