#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/23/2016"
"""
# ------------------------------------------------------------------------------
# Note: The following section refers to some online resources
# https://github.com/tflearn/tflearn/blob/master/examples/images/autoencoder.py
#-------------------------------------------------------------------------------

import warnings
warnings.simplefilter('ignore')

import numpy as np
import tensorflow as tf
import tflearn
import time

class Autoencoder(object):
    """
    self-defined class for Autoencoders using TFlearn
    """
    def __init__(self, layers=(784, 256, 64, 256, 784), init_w=None, init_b=None,
                 num_cores=4, gpu_memory_fraction=0.5):
        """ initialize parameters """
        self.n = len(layers)
        self.layers = layers
        self.init_w = init_w
        self.init_b = init_b
        self.num_cores = num_cores
        self.gpu_memory_fraction = gpu_memory_fraction
    # end __init__()

    def train(self, train_x, cv_x, test_x, n_epoch=10, batch_size=256):
        """ build and train the auto-encoder model """
        tflearn.init_graph(num_cores=self.num_cores, gpu_memory_fraction=self.gpu_memory_fraction)
        # build the encoder
        i = 0
        encoder = tflearn.input_data(shape=[None, self.layers[i]])
        i += 1
        while i <= self.n // 2:
            if self.init_w is None:
                encoder = tflearn.fully_connected(encoder, self.layers[i], activation='sigmoid')
            else:
                w = tf.constant(np.float32(self.init_w[i-1]))
                b = tf.constant(np.float32(self.init_b[i-1]))
                encoder = tflearn.fully_connected(encoder, self.layers[i], activation='sigmoid',
                                                  weights_init=w, bias_init=b)
            i += 1
        # end while

        # build the decoder
        if self.init_w is None:
            decoder = tflearn.fully_connected(encoder, self.layers[i], activation='sigmoid')
        else:
            w = tf.constant(np.float32(self.init_w[i-1]))
            b = tf.constant(np.float32(self.init_b[i-1]))
            decoder = tflearn.fully_connected(encoder, self.layers[i], activation='sigmoid',
                                              weights_init=w, bias_init=b)
        # end if

        i += 1
        while i < self.n:
            if self.init_w is None:
                decoder = tflearn.fully_connected(decoder, self.layers[i], activation='sigmoid')
            else:
                w = tf.constant(np.float32(self.init_w[i-1]))
                b = tf.constant(np.float32(self.init_b[i-1]))
                decoder = tflearn.fully_connected(decoder, self.layers[i], activation='sigmoid',
                                                  weights_init=w, bias_init=b)
            i += 1
        # end while

        # regression using mean-squared-error
        net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                                 loss='mean_square')

        # train the Auto-encoder model
        model = tflearn.DNN(net)
        model.fit(train_x, train_x, validation_set=(cv_x, cv_x),
                  n_epoch=n_epoch, batch_size=batch_size)

        # build a new model for encoder part
        encoder_model = tflearn.DNN(encoder, session=model.session)
        train_encoded = np.array(encoder_model.predict(train_x))
        if cv_x is not None:
            cv_encoded = np.array(encoder_model.predict(cv_x))
        test_encoded = np.array(encoder_model.predict(test_x))

        # build a new model for decoder part
        decoder_model = tflearn.DNN(decoder, session=model.session)
        train_decoded = np.array(decoder_model.predict(train_x))
        if cv_x is not None:
            cv_decoded = np.array(decoder_model.predict(cv_x))
        test_decoded = np.array(decoder_model.predict(test_x))

        print('Training process is finished !')
        tf.reset_default_graph()  # reset the Graph()
        return train_encoded, train_decoded, cv_encoded, cv_decoded, test_encoded, test_decoded
    # end train()

# end class Autoencoder()
