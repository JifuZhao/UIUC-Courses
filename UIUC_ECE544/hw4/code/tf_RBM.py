#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/03/2016"

The following section refers to some online resources
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

class RBM(object):
    """
    self-defined class for Restricted Boltzmann Machine (RBM) using TensorFlow
    """
    def __init__(self, visible_nodes, hidden_nodes, learning_rate, batch_size, n_iter, verbose=0):
        """ initialize the RBM """
        self.m = visible_nodes
        self.n = hidden_nodes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose

        self.tf_session = None

        self.W = None
        self.b = None
        self.c = None

        self.v0 = None
        self.h0 = None
        self.v1 = None

        self.W_updated = None
        self.b_updated = None
        self.c_updated = None


    def _initialize(self):
        """ function to initialize W, b and c """
        # initialize the parameters
        self.v0 = tf.placeholder(tf.float32, [None, self.m])
        self.h0 = tf.placeholder(tf.float32, [None, self.n])
        self.v1 = tf.placeholder(tf.float32, [None, self.m])

        self.W = tf.Variable(tf.truncated_normal((self.n, self.m), mean=0.0, stddev=0.01))
        self.b = tf.Variable(tf.truncated_normal([self.m], mean=0.0, stddev=0.1))
        self.c = tf.Variable(tf.truncated_normal([self.n], mean=0.0, stddev=0.1))

        # update v0, h0, v1, h1
        v0 = self.v0
        h0_prob = tf.nn.sigmoid(tf.matmul(v0, tf.transpose(self.W)) + self.c)
        h0 = self.sample_prob(h0_prob)

        v1_prob = tf.nn.sigmoid(tf.matmul(h0, self.W) + self.b)
        v1 = self.sample_prob(v1_prob)

        h1_prob = tf.nn.sigmoid(tf.matmul(v1, tf.transpose(self.W)) + self.c)
        h1 = self.sample_prob(h1_prob)

        # calculate the gradient
        # dW = tf.matmul(tf.transpose(h0_prob), v0) - tf.matmul(tf.transpose(h1_prob), v1) / self.batch_size
        dW = tf.matmul(tf.transpose(h0_prob), v0) - tf.matmul(tf.transpose(h1_prob), v1)
        db = tf.reduce_mean(v0 - v1, 0)
        dc = tf.reduce_mean(h0_prob - h1_prob, 0)

        # update parameters
        self.W_updated = self.W.assign_add(self.learning_rate * dW)
        self.b_updated = self.b.assign_add(self.learning_rate * db)
        self.c_updated = self.c.assign_add(self.learning_rate * dc)


    def train(self, X, test=None):
        """ function to train the RBM """
        t0 = time.time()
        N_sample, _ = X.shape

        # get the batches
        if N_sample % self.batch_size == 0:
            N_batch = N_sample // self.batch_size
        else:
            N_batch = N_sample // self.batch_size + 1

        # initialize the parameters
        self._initialize()
        init_op = tf.initialize_all_variables()

        # begin training
        with tf.Session() as self.tf_session:
            self.tf_session.run(init_op)
            for iteration in range(1, self.n_iter + 1):
                for i in range(N_batch):
                    v = X[i*self.batch_size: (i+1)*self.batch_size, :]
                    self.tf_session.run([self.W_updated, self.b_updated, self.c_updated],
                                        feed_dict={self.v0: v})

            t = np.round(time.time() - t0, 2)
            print('Reach the ' + str(iteration) + ' th iteration, total time ' + str(t) + 's !')

            # return trained parameters
            return self.W.eval(), self.b.eval(), self.c.eval()


    def sample_prob(self, probs):
        """ function for Bernoulli sample """
        # rand = tf.random_uniform(probs.get_shape())
        rand = tf.random_uniform(tf.shape(probs))
        return tf.nn.relu(tf.sign(probs - rand))


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
        tmp[:, number] = (label[:, 0] == number)
    tmp = tmp.astype(int)

    return tmp
