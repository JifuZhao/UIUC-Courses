#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "12/05/2016"
"""

import warnings
import numpy as np
import time
from k_means import kMeans

warnings.simplefilter('ignore')


def spectralClustering(X, K, tol=10e-5, random_state=None, verbose=True):
    """ function to implement the spectral clustering algorithm """
    t0 = time.time()

    N, d = X.shape
    W = np.zeros((N, N))  # adjacency matrix W
    for i in range(N):
        distance = np.sqrt(np.sum((X - X[i, :])**2, axis=1))
        W[:, i] = distance

    diag = np.sum(W, axis=1)
    D = np.diag(diag)  # diagnoal matrix D
    L = D - W  # Laplacian matrix L
    eigvals, U = np.linalg.eigh(L)
    U = U[:, :K]  # first K eigenvectors

    # call k-means for clustering
    _, C, _ = kMeans(U, K, tol=tol, random_state=random_state, verbose=False)

    if verbose is True:
        t = np.round(time.time() - t0, 4)
        print('Spectral Clustering finished in ' + str(t) + 's')

    return W, U, C


def main():
    """ implement the spectral clustering on real-world data """
    print('test case !')


if __name__ == '__main__':
    main()
