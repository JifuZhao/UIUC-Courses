#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/15/2016"

Note: this implementation followed some online tutorials:
https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html
https://tensorhub.com/aymericdamien/tensorflow-rnn
https://github.com/tflearn/tflearn/blob/master/examples/images/rnn_pixels.py
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import time
import tensorflow as tf

class RNN(object):
    """ One-layer RNN """

    def __init__(self, category, learning_rate, max_iters, batch_size, n_input,
                 n_steps, n_hidden, n_classes, regression='logistic'):
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
        self.regression = regression  # logistic or linear type

        # keep recording the training and testing accuracy
        self.train_acc = []
        self.cv_acc = []

        # define input and outputs
        self.X = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

        # define W and b for final classification f(W * h + b)
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

        # logistic or linear activation
        if self.regression == 'logistic':
            return tf.nn.softmax(tf.matmul(outputs[-1], self.W) + self.b)
        elif self.regression == 'linear':
            return tf.matmul(outputs[-1], self.W) + self.b
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
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        outputs, states = tf.nn.rnn(cell, X, dtype=tf.float32)

        # logistic or linear activation
        if self.regression == 'logistic':
            return tf.nn.softmax(tf.matmul(outputs[-1], self.W) + self.b)
        elif self.regression == 'linear':
            return tf.matmul(outputs[-1], self.W) + self.b
    # end LSTM()


    def train(self, mnist, order='C', frequence=10, size=None, show_frequence=None):
        """ function to train the model, restricted to MNIST dataset """
        t0 = time.time()

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

        # Launch the graph
        with tf.Session() as sess:
            sess.run(initiator)
            # Keep training until reach max iterations
            for i in range(1, self.max_iters + 1):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)

                # reshape the given data into (batch_size, n_steps, n_input)
                batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input), order=order)

                # begin training
                sess.run(optimizer, feed_dict={self.X: batch_x, self.Y: batch_y})

                # keep recording the current accuracy
                if i % frequence == 0:
                    # get all training and testing data
                    if size is not None:
                        train_imgs, train_label = mnist.train.next_batch(size)
                        cv_imgs, cv_label = mnist.validation.next_batch(size)
                        # reshape
                        train_imgs = train_imgs.reshape((len(train_imgs), self.n_steps, self.n_input), order=order)
                        cv_imgs = cv_imgs.reshape((len(cv_imgs), self.n_steps, self.n_input), order=order)
                    else:
                        train_imgs = batch_x
                        train_label = batch_y
                        cv_imgs, cv_label = mnist.validation.next_batch(self.batch_size)
                        cv_imgs = cv_imgs.reshape((self.batch_size, self.n_steps, self.n_input), order=order)

                    tmp_train_acc = sess.run(accuracy, feed_dict={self.X: train_imgs,
                                                                  self.Y: train_label})
                    tmp_cv_acc = sess.run(accuracy, feed_dict={self.X: cv_imgs,
                                                               self.Y: cv_label})
                    self.train_acc.append(tmp_train_acc)
                    self.cv_acc.append(tmp_cv_acc)

                    if (show_frequence is not None) and (i % show_frequence == 0):
                        print("Iteration " + str(i) + \
                              ", Training Accuracy= " + "{:.5f}".format(tmp_train_acc) +\
                              ", Validation Accuracy= " + "{:.5f}".format(tmp_cv_acc))

            # evaluate the performance on the whole dataset
            all_train_imgs = mnist.train.images
            all_train_label = mnist.train.labels
            all_test_imgs = mnist.test.images
            all_test_label = mnist.test.labels

            # reshape
            all_train_imgs = all_train_imgs.reshape((len(all_train_imgs), self.n_steps, self.n_input), order=order)
            all_test_imgs = all_test_imgs.reshape((len(all_test_imgs), self.n_steps, self.n_input), order=order)

            # calculate the accuracy
            final_train_acc = sess.run(accuracy, feed_dict={self.X: all_train_imgs,
                                                            self.Y: all_train_label})
            final_test_acc = sess.run(accuracy, feed_dict={self.X: all_test_imgs,
                                                            self.Y: all_test_label})

        # reset the graph to avoid potential problems
        tf.reset_default_graph()

        t = np.round(time.time() - t0)
        print('\n')
        print('Training is finished in', t, 's !')
        print('Final Training Accuracy: ' + '{:.5f}'.format(final_train_acc))
        print('Final Testing Accuracy:  ' + '{:.5f}'.format(final_test_acc))
    # end train()


    def get_params(self):
        """ function to get all parameters """
        return self.train_acc, self.cv_acc
    # end get_params()

# end class RNN()
