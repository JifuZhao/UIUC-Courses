#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "09/24/2016"
"""

import numpy as np

class PCA(object):
    """ class PCA is used to perform principal component analysis """
    def __init__(self, n):
        self.n = n
        self.mean = None
        self.component = None
        self.variance = None


    def fit(self, X):
        """ perform PCA through SVD
            X has dimension n * m
            where n is number of records, m is number of features
        """
        self.mean = np.mean(X, axis=0)
        centeredX = X - self.mean
        U, S, V = np.linalg.svd(centeredX)
        self.variance = S ** 2
        self.component = V

        return self.variance[:self.n], self.mean, self.component[:, :self.n]


def main():
    X = np.random.random((10, 4))
    pca = PCA(n=4)
    variance, mean, eigenvector = pca.fit(X)
    print('Variance: \t', variance)
    print('Mean: \t', mean)
    print('Eigenvector: \t', eigenvector)


if __name__ == '__main__':
    main()
