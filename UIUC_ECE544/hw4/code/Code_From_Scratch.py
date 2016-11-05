#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/04/2016"

"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

from RBM import RBM


# load MNIST dataset
mndata = MNIST('./data/')

train_img, train_label = mndata.load_training()
train_img = np.array(train_img)

test_img, test_label = mndata.load_testing()
test_img = np.array(test_img)

# transform input data into binary values
data = (train_img > 0).astype(float)
test_data = (test_img > 0).astype(float)


# # RBM
# build the Restricted Boltzmann Machine
model = RBM(hidden_nodes=200, learning_rate=0.1, batch_size=1, n_iter=3, verbose=1)

model.train(data)
W, b, c = model.get_params()

# visualize the learned weights
imgs = np.reshape(W[:200, :], (200, 28, 28))

fig, ax = plt.subplots(nrows=8, ncols=8)
for i in range(8):
    for j in range(8):
        ax[i, j].imshow(imgs[i * 8 + j, :, :], cmap=plt.cm.gray_r, interpolation='none')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].axis('image')
plt.tight_layout(pad=0.5, h_pad=0, w_pad=0, rect=None)

fig.savefig('./result/filter.pdf')
fig.savefig('./result/filter.png', dpi=300)
plt.show()


# build the Restricted Boltzmann Machine
model = RBM(hidden_nodes=200, learning_rate=0.001, batch_size=10, n_iter=3, verbose=1)

model.train(data)
W, b, c = model.get_params()

# visualize the learned weights
imgs = np.reshape(W[:200, :], (200, 28, 28))

fig, ax = plt.subplots(nrows=8, ncols=8)
for i in range(8):
    for j in range(8):
        ax[i, j].imshow(imgs[i * 8 + j, :, :], cmap=plt.cm.gray_r, interpolation='none')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].axis('image')
plt.tight_layout(pad=0.5, h_pad=0, w_pad=0, rect=None)

fig.savefig('./result/filter_batch.pdf')
fig.savefig('./result/filter_batch.png', dpi=300)
plt.show()




