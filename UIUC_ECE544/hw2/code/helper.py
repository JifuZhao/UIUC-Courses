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
            matrix = f.readlines()
        # skip the data that has less than 70 frames
        if len(matrix) < 70:
            continue
        tmp = []
        indicator = True
        for data in matrix[:70]:
            data = data.split()
            if 'NaN' in data or '-Inf' in data or 'Inf' in data:
                indicator = False
                break
            data = list(map(float, data))
            tmp += data
        if indicator == True:
            featureSet.append(tmp)
            labelSet.append(int(label))

    labels = np.array([labelSet], dtype=int).T
    features = np.array(featureSet)

    return features, labels


def oneHotEncoder(label, n):
    """ One-Hot-Encoder for n class case """
    tmp = np.zeros((len(label), n))
    for number in range(n):
        tmp[:, number] = (label[:, 0] == number)
    tmp = tmp.astype(int)

    return tmp
