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

def PCA(X, standardize=False):
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
    projection = np.dot(X, U)

    return S, U, projection


def kernelPCA(X, kernel='polynomial', degree=2, gamma=1):
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
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # normalize the eigenvectors
    eigvecs = eigvecs / np.sqrt(eigvals)

    # compute the projection
    projection = K.dot(eigvecs)


    return eigvals, eigvecs, projection
    
    
## Helper function for better visualization
def showConfusionMatrix(matrix, title, label, fontsize=10):
    """ function to show the confusion matrix"""

    fig = plt.figure()
    img = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(img)

    n = len(label)
    plt.xticks(np.arange(n), label)
    plt.yticks(np.arange(n), label)

    for i, j in [(row, col) for row in range(n) for col in range(n)]:
        plt.text(j, i, matrix[i, j], horizontalalignment="center", fontsize=fontsize)

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig


def oneHotEncoder(label, n):
    """ One-Hot-Encoder for n class case """
    tmp = np.zeros((len(label), n))
    for number in range(n):
        tmp[:, number] = (label == number)
    tmp = tmp.astype(int)

    return tmp
