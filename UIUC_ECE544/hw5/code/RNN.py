#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "mm/dd/2016"

Note: this implementation followed the online tutorial:
https://tensorhub.com/aymericdamien/tensorflow-rnn
https://github.com/tflearn/tflearn/blob/master/examples/images/rnn_pixels.py
https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import tensorflow as tf

class RNN(object):
    """ One-layer RNN """

    def __init__(self, category, learning_rate, max_iters, batch_size, n_input, n_steps, n_hidden, n_classes):
        """ initialize all parameters """
        # define parameters
        self.category = category  # basicRNN or LSTM
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size

        self.n_input = n_input
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.train_acc = []
        self.test_acc = []

        # define input and outputs
        self.X = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

        # define W and b for final classification
        # prediction = softmax(W * h + b)
        self.W = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
        self.b = tf.Variable(tf.truncated_normal([n_classes]))
    # end __init__()


    def basicRNN(self, X):
        """ function to build the basic RNN model """

        # reshape the input to fit the requrement of RNN
        # according to https://github.com/tensorflow/tensorflow/blob/master/
        # tensorflow/g3doc/api_docs/python/functions_and_classes/shard0/tf.nn.rnn.md
        X = tf.transpose(X, [1, 0, 2])
        # reshaping to (n_steps * batch_size, n_input)
        X = tf.reshape(X, [-1, self.n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        X = tf.split(0, self.n_steps, X)

        # create the basic RNN cell
        cell = tf.nn.rnn_cell.BasicRNNCell(self.n_hidden)
        outputs, states = tf.nn.rnn(cell, X, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.nn.softmax(tf.matmul(outputs[-1], self.W) + self.b)
    # end basicRNN()


    def LSTM(self, X):
        """ function to build the LSTM model """

        # reshape the input to fit the requrement of RNN
        # according to https://github.com/tensorflow/tensorflow/blob/master/
        # tensorflow/g3doc/api_docs/python/functions_and_classes/shard0/tf.nn.rnn.md
        X = tf.transpose(X, [1, 0, 2])
        # reshaping to (n_steps * batch_size, n_input)
        X = tf.reshape(X, [-1, self.n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        X = tf.split(0, self.n_steps, X)

        # create the basic RNN cell
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        outputs, states = tf.nn.rnn(cell, X, dtype=tf.float32)

        # using the last frame for calssification
        return tf.nn.softmax(tf.matmul(outputs[-1], self.W) + self.b)
    # end LSTM()


    def train(self, mnist, order='C', frequence=10, size=300):
        """ function to train the model, restricted to MNIST dataset """

        # get the prediction
        if self.category == 'basicRNN':
            prediction = self.basicRNN(self.X)
        elif self.category == 'LSTM':
            prediction = self.LSTM(self.X)

        # optimize the cross_entropy
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

        # calculate the predicted label and accuracy
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initializing the variables
        initiator = tf.initialize_all_variables()

        # get all training and testing data
        if size is not None:
            train_imgs = mnist.train.images[:size]
            train_label = mnist.train.labels[:size]
            test_imgs = mnist.test.images[:size]
            test_label = mnist.test.labels[:size]
        else:
            train_imgs = mnist.train.images
            train_label = mnist.train.labels
            test_imgs = mnist.test.images
            test_label = mnist.test.labels
            
        # reshape
        train_imgs = train_imgs.reshape((len(train_imgs), self.n_steps, self.n_input), order='C')
        test_imgs = test_imgs.reshape((len(test_imgs), self.n_steps, self.n_input), order='C')

        # Launch the graph
        with tf.Session() as sess:
            sess.run(initiator)
            # Keep training until reach max iterations
            for i in range(1, self.max_iters):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)

                # reshape the given data into (batch_size, n_steps, n_input)
                batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input), order='C')

                # begin training
                sess.run(optimizer, feed_dict={self.X: batch_x, self.Y: batch_y})

                # keep recording the current accuracy
                if i % frequence == 0:
                    tmp_train_acc = sess.run(accuracy, feed_dict={self.X: train_imgs,
                                                                  self.Y: train_label})
                    tmp_test_acc = sess.run(accuracy, feed_dict={self.X: test_imgs,
                                                                 self.Y: test_label})
                    self.train_acc.append(tmp_train_acc)
                    self.test_acc.append(tmp_test_acc)

        # reset the graph to avoid potential problems
        tf.reset_default_graph()
        print("Training is finished !")
        print('Final Training accuracy:\t', np.round(self.train_acc[-1], 5))
        print('Final Testing accuracy:\t', np.round(self.test_acc[-1], 5))
    # end train()


    def get_params(self):
        """ function to get all parameters """
        return self.train_acc, self.test_acc
    # end get_params()

# end class RNN()
