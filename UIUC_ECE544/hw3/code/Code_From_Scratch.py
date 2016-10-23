#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "10/16/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from EM import EM


# # Corgi image
# read the image
img = plt.imread('./corgi.png')[:, :, :3]

# reshape the input vector
x = np.reshape(img, (-1, 3))
x = x / np.max(x)


# ## 3 Mixtures
np.random.seed(2015)  # 7, 10, 12, 30, 70
# train the EM model
m = 3
em = EM(m=m, threshold=0.0001, maxIter=50)
em.train(x)

# get all parameters
w, mu, sigma, logLikelihood = em.get_params()
# get the label
label = em.get_label()

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, m+1):
    test[label == i] = mu[i - 1]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 3')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/corgi_3.png', dpi=300)
plt.show()

print('w:\n', np.round(w, 5))
print('\nmu:\n', np.round(mu, 5))
print('\nsigma:\n', np.round(sigma, 5))


# ## 5 mixtures
np.random.seed(200)  # 6, 16, 200, 2000
# train the EM model
m = 5
em = EM(m=m, threshold=0.0001, maxIter=50)
em.train(x)

# get all parameters
w, mu, sigma, logLikelihood = em.get_params()
# get the label
label = em.get_label()

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, m+1):
    test[label == i] = mu[i - 1]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 5')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/corgi_5.png', dpi=300)
plt.show()


# ## 10 mixtures
np.random.seed(5)  # 1, 2, 2000
# train the EM model
m = 10
em = EM(m=m, threshold=0.001, maxIter=30)
em.train(x)

# get all parameters
w, mu, sigma, logLikelihood = em.get_params()
# get the label
label = em.get_label()

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, m+1):
    test[label == i] = mu[i - 1]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 10')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/corgi_10.png', dpi=300)
plt.show()


# # Self figure
# read the image
img = plt.imread('./self.jpg')

# reshape the input vector
x = np.reshape(img, (-1, 3))
x = x / np.max(x)


# ## 3 Mixtures
np.random.seed(7)  # 7,
# train the EM model
m = 3
em = EM(m=m, threshold=0.0001, maxIter=50)
em.train(x)

# get all parameters
w, mu, sigma, logLikelihood = em.get_params()
# get the label
label = em.get_label()

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, m+1):
    test[label == i] = mu[i - 1]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 3')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/self_3.png', dpi=300)
plt.show()

print('w:\n', np.round(w, 5))
print('\nmu:\n', np.round(mu, 5))
print('\nsigma:\n', np.round(sigma, 5))


# ## 5 Mixtures
np.random.seed(10)  # 1, 2, 7, 8, 10
# train the EM model
m = 5
em = EM(m=m, threshold=0.0001, maxIter=50)

em.train(x)

# get all parameters
w, mu, sigma, logLikelihood = em.get_params()
# get the label
label = em.get_label()

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, m+1):
    test[label == i] = mu[i - 1]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 5')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/self_5.png', dpi=300)
plt.show()


# ## 10 Mixtures
np.random.seed(1)  # 1, 5, 7, 9
# train the EM model
m = 10
em = EM(m=m, threshold=0.001, maxIter=30)
em.train(x)

# get all parameters
w, mu, sigma, logLikelihood = em.get_params()
# get the label
label = em.get_label()

# reconstruct the image
test = copy.deepcopy(x)
for i in range(1, m+1):
    test[label == i] = mu[i - 1]

fig = plt.figure()
plt.imshow(np.reshape(test, img.shape))
plt.title('Reconstruction with m = 10')
plt.xticks([])
plt.yticks([])
fig.savefig('./result/self_10.png', dpi=300)
plt.show()
