#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    __author__      = "Jifu Zhao"
    __email__       = "jzhao59@illinois.edu"
    __date__        = "12/08/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from k_means import kMeans
from k_centers import kCenters
from single_swap import singleSwap
from spectral_clustering import spectralClustering
from EM import EM

# read the data
clustering = pd.read_csv('./data/clustering.csv', header=None).values
bigClustering = pd.read_csv('./data/bigClusteringData.csv', header=None).values

# define colors
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500']


# # 0. Plot the original distribution
# plot the clustered data
fig, ax = plt.subplots()
ax.plot(clustering[:, 0], clustering[:, 1], 'g.', markersize=4)
ax.set_title('Scatter Plot of clustering.txt')
ax.legend(fontsize=8, loc='best')
fig.savefig('./result/pdf/clustering_scatter.pdf')
fig.savefig('./result/clustering_scatter.png', dpi=300)
plt.show()

# plot the clustered data
fig, ax = plt.subplots()
ax.plot(bigClustering[:, 0], bigClustering[:, 1], 'g.', markersize=3)
ax.set_title('Scatter Plot of bigClusteringData.txt')
ax.legend(fontsize=8, loc='best')
fig.savefig('./result/pdf/bigClustering_scatter.pdf')
fig.savefig('./result/bigClustering_scatter.png', dpi=300)
plt.show()


# # I. K-Means Algorithm
# Test of convergence with different tols
print('clustering.txt')
X = clustering
K = 3

for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
    Q, C, D = kMeans(X, K, tol=tol, random_state=None, verbose=False)
    print('tol is', tol, '\t', 'D is', D)

print('\n')
print('bigClusteringData.txt')
X = bigClustering
K = 3

for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
    Q, C, D = kMeans(X, K, tol=tol, random_state=None, verbose=False)
    print('tol is', tol, '\t', 'D is', D)


# ------------------------------------------------------------------------
# clustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
tol = 1e-5
X = clustering

for K in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        Q, C, D = kMeans(X, K, tol=tol, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[best_C == i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.5, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(best_Q[:, 0], best_Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('K-Means with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/clustering_kMeans_' + str(K) + '.pdf')
    fig.savefig('./result/clustering_kMeans_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(2, 11), loss, 'go-')
ax.set_title('D vs. K for K-Means Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_clustering_kMeans.pdf')
fig.savefig('./result/loss_clustering_kMeans.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# ------------------------------------------------------------------------
# bigClustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
tol = 1e-5
X = bigClustering

for K in [3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        Q, C, D = kMeans(X, K, tol=tol, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[best_C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.5, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(best_Q[:, 0], best_Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('K-Means with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/bigClustering_kMeans_' + str(K) + '.pdf')
    fig.savefig('./result/bigClustering_kMeans_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(3, 11), loss, 'go-')
ax.set_title('D vs. K for K-Means Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_bigClustering_kMeans.pdf')
fig.savefig('./result/loss_bigClustering_kMeans.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# # II. Greedy K-Centers Algorithm

# ------------------------------------------------------------------------
# clustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
X = clustering

for K in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        Q, C, D, _ = kCenters(X, K, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.5, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('K-Centers with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/clustering_kCenter_' + str(K) + '.pdf')
    fig.savefig('./result/clustering_kCenter_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(2, 11), loss, 'go-')
ax.set_title('D vs. K for K-Centers Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_clustering_kCenter.pdf')
fig.savefig('./result/loss_clustering_kCenter.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# ------------------------------------------------------------------------
# bigClustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
X = bigClustering

for K in [3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        Q, C, D, _ = kCenters(X, K, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.4, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('K-Centers with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/bigClustering_kCenter_' + str(K) + '.pdf')
    fig.savefig('./result/bigClustering_kCenter_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(3, 11), loss, 'go-')
ax.set_title('D vs. K for K-Centers Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_bigClustering_kCenter.pdf')
fig.savefig('./result/loss_bigClustering_kCenter.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# # III. Single-Swap Algorithm

# ------------------------------------------------------------------------
# clustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
tau = 0.05
X = clustering

for K in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        Q, C, D = singleSwap(X, K, tau=tau, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.5, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('Single-Swap with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/clustering_singleSwap_' + str(K) + '.pdf')
    fig.savefig('./result/clustering_singleSwap_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(2, 11), loss, 'go-')
ax.set_title('D vs. K for Single-Swap Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_clustering_singleSwap.pdf')
fig.savefig('./result/loss_clustering_singleSwap.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# ------------------------------------------------------------------------
# bigClustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
tau = 0.05
X = bigClustering

for K in [3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        Q, C, D = singleSwap(X, K, tau=tau, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.4, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('Single-Swap with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/bigClustering_singleSwap_' + str(K) + '.pdf')
    fig.savefig('./result/bigClustering_singleSwap_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(3, 11), loss, 'go-')
ax.set_title('D vs. K for Single-Swap Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_bigClustering_singleSwap.pdf')
fig.savefig('./result/loss_bigClustering_singleSwap.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# # IV. Spectral Clustering Algorithm

# ------------------------------------------------------------------------
# clustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 20
X = clustering

for K in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        W, U, Q, C, D = spectralClustering(X, K, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.6, markersize=3, label='Cluster ' + str(i+1))
#     ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('Spectral Clustering with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/clustering_spectral_' + str(K) + '.pdf')
    fig.savefig('./result/clustering_spectral_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(2, 11), loss, 'go-')
ax.set_title('D vs. K for Spectral Clustering Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_clustering_spectral.pdf')
fig.savefig('./result/loss_clustering_spectral.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# ------------------------------------------------------------------------
# bigClustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 20
X = bigClustering

for K in [3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        W, U, Q, C, D = spectralClustering(X, K, random_state=None, verbose=False)
        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.4, markersize=3, label='Cluster ' + str(i+1))
#     ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('Spectral Clustering with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/bigClustering_spectral_' + str(K) + '.pdf')
    fig.savefig('./result/bigClustering_spectral_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(3, 11), loss, 'go-')
ax.set_title('D vs. K for Spectral Clustering Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_bigClustering_spectral.pdf')
fig.savefig('./result/loss_bigClustering_spectral.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])


# # V. EM Algorithm

# ------------------------------------------------------------------------
# clustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
threshold = 1e-7
X = clustering

for K in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        em = EM(m=K, threshold=threshold, random_state=None, maxIter=500)
        em.train(X, verbose=False)  # train the EM model

        # get the label
        C = em.get_label()  # label for each x
        Q = np.array(em.mu)  # cluster centers
        D = em.D

        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.5, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('EM with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/clustering_EM_' + str(K) + '.pdf')
    fig.savefig('./result/clustering_EM_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(2, 11), loss, 'go-')
ax.set_title('D vs. K for EM Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_clustering_EM.pdf')
fig.savefig('./result/loss_clustering_EM.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])



# ------------------------------------------------------------------------
# bigClustering.txt
# ------------------------------------------------------------------------
loss = []
index = []
iteration = 50
threshold = 1e-7
X = bigClustering

for K in [3, 4, 5, 6, 7, 8, 9, 10]:
    best_D = None
    best_C = None
    best_Q = None
    for i in range(iteration):
        em = EM(m=K, threshold=threshold, random_state=None, maxIter=500)
        em.train(X, verbose=False)  # train the EM model

        # get the label
        C = em.get_label()  # label for each x
        Q = np.array(em.mu)  # cluster centers
        D = em.D

        if best_D is None:
            best_D = D
            best_C = C
            best_Q = Q
        else:
            if D < best_D:
                best_D = D
                best_C = C
                best_Q = Q

    loss.append(best_D)
    index.append(best_C)

    # plot the clustered data
    fig, ax = plt.subplots()
    for i in range(K):
        tmp = X[C==i, :]
        ax.plot(tmp[:, 0], tmp[:, 1], '.', color=colors[i],
                alpha=0.4, markersize=3, label='Cluster ' + str(i+1))
    ax.plot(Q[:, 0], Q[:, 1], '*', color='k', markersize=10, label='Centroids')
    ax.set_title('EM with K=' + str(K))
    ax.legend(fontsize=8, loc='best')
    fig.savefig('./result/pdf/bigClustering_EM_' + str(K) + '.pdf')
    fig.savefig('./result/bigClustering_EM_' + str(K) + '.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# plot the change of D versus K
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(3, 11), loss, 'go-')
ax.set_title('D vs. K for EM Algorithm')
ax.set_xlabel('K')
ax.set_ylabel('D')
ax.grid('on')
fig.savefig('./result/pdf/loss_bigClustering_EM.pdf')
fig.savefig('./result/loss_bigClustering_EM.png', dpi=300)
plt.show()
print('D\t', loss)
print('Index')
for i in index:
    print(i[:20])
