#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/01/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
from scipy.linalg import eigh

from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel

def PCA(X, n_component, standardize=False):
    """ self-defined PCA function """

    # de-mean (and standardize) the input X
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = X - mean
    if standardize == True:
        X = X / std

    # calculate the covariance matrix
    n = len(X)
    cov = np.dot(X.T, X) / n
    U, S, V = np.linalg.svd(cov)

    # compute the projection
    projection = np.dot(X, U[:, :n_component])

    return S, U[:, :n_component], projection


def kernelPCA(X, n_component, kernel='polynomial', degree=2, gamma=1):
    """ self-defined Kernel PCA function
        can implement polynomial and rbf kernel
    """
    # compute the kernel matrix K
    if kernel == 'polynomial':
        K = polynomial_kernel(X, degree=degree, gamma=gamma)
    elif kernel == 'rbf':
        K = rbf_kernel(X, gamma=gamma)
    else:
        print('Only support polynomial and rbf kernel !')
        return

    # center the kernel K
    n = len(K)
    one = np.ones((n, n)) / n
    K = K - one.dot(K) - K.dot(one) + one.dot(K).dot(one)

    # calculate the eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(K)
    eigvals = eigvals[::-1][:n_component]
    eigvecs = eigvecs[:, ::-1][:, :n_component]

    # normalize the eigenvectors
    eigvecs = eigvecs / np.sqrt(eigvals)

    # compute the projection
    projection = K.dot(eigvecs)


    return eigvals, eigvecs, projection
