#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "12/05/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def kMeans(X, K, tol=0.00001, random_state=None, verbose=True):
    """ function to implement the Lloyd's algorithm for k-means problem """
    np.random.seed(random_state)
    t0 = time.time()

    N, d = X.shape  # number of observations and dimensions
    index = np.random.choice(range(N), size=K, replace=False)
    Y = X[index, :]  # initial k centers
    C = np.zeros(N)
    D = 100
    count = 0
    loss = 100

    while loss > tol:
        D0 = D
        for i in range(N):
            # assign centers to ith data
            C[i] = np.argmin(np.sum((Y - X[i, :]) ** 2, axis=1))

        D = 0
        # re-compute the new centers
        for j in range(K):
            Y[j, :] = np.mean(X[C==j, :], axis=0)
            D += np.sum(np.sqrt(np.sum((X[C==j, :] - Y[j, :]) ** 2, axis=1)))

        # compute the average loss
        D = D / N
        loss = abs(D - D0)
        count += 1

    if verbose == True:
        t = np.round(time.time() - t0, 4)
        print('K-Means is finished in ' + str(t) + 's, ' + str(count) + ' iterations')

    return Y, C, D


def main():
    """ implement the k-means algorithm on real-world data """
    print('test case !')

if __name__ == '__main__':
    main()
