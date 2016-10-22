#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "10/22/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

import tensorflow as tf
from tensorflow.contrib.factorization.python.ops.gmm import GMM
from tensorflow.contrib.factorization.python.ops.kmeans import KMeansClustering as KMeans

# # Original Figure
# read the image
img = plt.imread('./corgi.png')[:, :, :3]
# reshape the input vector
x = np.reshape(img, (-1, 3))
x = x / np.max(x)

# ## 3 clusters
random_seed = 3
# train the GMM model
m = 3
gmm = GMM(num_clusters=m, random_seed=random_seed, initial_clusters='random',
          covariance_type='full', verbose=False)
gmm.fit(x)
label = gmm.predict(x)[0, :] + 1

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, 4):
    test[label == i] = gmm.clusters()[i - 1, :]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 3')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/corgi_tf_3.png', dpi=300)
plt.show()

print('Weights:\n', gmm.get_variable_value('Variable'))
print('Mean:\n', gmm.clusters())
print('\nCovariance:\n', gmm.covariances())


# ## 5 Clusters
random_seed = 10
# train the GMM model
m = 5
gmm = GMM(num_clusters=m, random_seed=random_seed, initial_clusters='random',
          covariance_type='full', verbose=False)
gmm.fit(x)
label = gmm.predict(x)[0, :] + 1

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, 4):
    test[label == i] = gmm.clusters()[i - 1, :]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 5')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/corgi_tf_5.png', dpi=300)
plt.show()


# ## 10 Clusters
random_seed = 2
# train the GMM model
m = 10
gmm = GMM(num_clusters=m, random_seed=random_seed, initial_clusters='random',
          covariance_type='full', verbose=False)
gmm.fit(x)
label = gmm.predict(x)[0, :] + 1

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, 4):
    test[label == i] = gmm.clusters()[i - 1, :]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 10')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/corgi_tf_10.png', dpi=300)
plt.show()


# # Self Image
# read the image
img = plt.imread('./self.jpg') / 255
# reshape the input vector
x = np.reshape(img, (-1, 3))
x = x / np.max(x)

# ## 3 Mixtures
random_seed = 5
# train the GMM model
m = 3
gmm = GMM(num_clusters=m, random_seed=random_seed, initial_clusters='random',
          covariance_type='full', verbose=False)
gmm.fit(x)
label = gmm.predict(x)[0, :] + 1

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, 4):
    test[label == i] = gmm.clusters()[i - 1, :]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 3')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/self_tf_3.png', dpi=300)
plt.show()

print('Weights:\n', gmm.get_variable_value('Variable'))
print('Mean:\n', gmm.clusters())
print('\nCovariance:\n', gmm.covariances())


# ## 5 Mixtures
random_seed = 7
# train the GMM model
m = 5
gmm = GMM(num_clusters=m, random_seed=random_seed, initial_clusters='random',
          covariance_type='full', verbose=False)
gmm.fit(x)
label = gmm.predict(x)[0, :] + 1

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, 4):
    test[label == i] = gmm.clusters()[i - 1, :]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 5')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/self_tf_5.png', dpi=300)
plt.show()


# ## 10 Mixtures
random_seed = 1
# train the GMM model
m = 10
gmm = GMM(num_clusters=m, random_seed=random_seed, initial_clusters='random',
          covariance_type='full', verbose=False)
gmm.fit(x)
label = gmm.predict(x)[0, :] + 1

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, 4):
    test[label == i] = gmm.clusters()[i - 1, :]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 10')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/self_tf_10.png', dpi=300)
plt.show()
