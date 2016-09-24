#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "09/21/2016"
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

def readFile(labelPath, featureDir='./data/'):
    """ labelPath is the path for label file
            Ex. labelPath = "./data/train/lab/hw1train_labels.txt"
        featureDir is the main directory for dataset
            Ex. featureDir = "./data/"

        return: features and labels
    """
    # read the labels and corresponding path
    with open(labelPath) as f:
        lines = f.readlines()

    labelSet = []
    featureSet = []
    for line in lines:
        label, fname = line.split()
        with open(featureDir + fname) as f:
            line = f.readline().split()
            # skip the corrupted data
            if 'NaN' in line or '-Inf' in line or 'Inf' in line:
                continue
            else:
                line = list(map(float, line))
                featureSet.append(line)
                labelSet.append(int(label))

    labels = np.array([labelSet], dtype=int).T
    features = np.array(featureSet)

    return features, labels


def addBias(data):
    """ function to add bias item to the data as the first column """
    n = len(data)
    data = np.append(np.array([np.ones(n)]).T, data, axis=1)
    return data


def oneHotEncoder(label):
    """ One-Hot-Encoder for two class case """
    tmp = np.zeros((len(label), 2))
    tmp[:, 0] = (label[:, 0] == 0)
    tmp[:, 1] = (label[:, 0] == 1)
    tmp = tmp.astype(int)

    return tmp
