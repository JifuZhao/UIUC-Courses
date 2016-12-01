#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "11/20/2016"
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import tensorflow as tf
import tflearn

def classification(train_x, train_y, test_x, test_y, dim_input, dim_output,
                   cv_x=None, cv_y=None, n_epoch=10, njobs=4, fraction=0.5):
    """ function to build a one-layer neural network for classification """

    # Classification using tflearn
    with tf.Graph().as_default():
        tflearn.init_graph(num_cores=njobs, gpu_memory_fraction=fraction)

        # build the one-layer fully connected neural network
        net = tflearn.input_data(shape=[None, dim_input])
        net = tflearn.fully_connected(net, dim_output, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

        # fit the model
        model = tflearn.DNN(net)
        if cv_x is None:
            validation_set = None
        else:
            validation_set = (cv_x[:, :dim_input], cv_y)
        model.fit(train_x[:, :dim_input], train_y, validation_set=validation_set,
                  n_epoch=n_epoch)

        # predict on the training and test dataset
        train_prediction = np.array(model.predict(train_x[:, :dim_input]))
        test_prediction = np.array(model.predict(test_x[:, :dim_input]))

    # get the label
    train_pred_label = np.argmax(train_prediction, axis=1)
    test_pred_label = np.argmax(test_prediction, axis=1)

    train_true_label = np.argmax(train_y, axis=1)
    test_true_label = np.argmax(test_y, axis=1)

    # calculate the training and testing accuracy
    train_acc = np.sum(train_pred_label == train_true_label) / len(train_true_label)
    test_acc = np.sum(test_pred_label == test_true_label) / len(test_true_label)

    return train_acc, test_acc
# end classification()


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
# end showConfusionMatrix()


def oneHotEncoder(label, n):
    """ One-Hot-Encoder for n class case """
    tmp = np.zeros((len(label), n))
    for number in range(n):
        tmp[:, number] = (label[:, 0] == number)
    tmp = tmp.astype(int)
# end oneHotEncoder()
    return tmp
