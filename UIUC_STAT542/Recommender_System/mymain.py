#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd

from surprise import KNNWithMeans
from surprise import SVD
from surprise import Dataset
from surprise import Reader

warnings.filterwarnings('ignore')

# Start of the algorithm
# # Read data

# build the reader and read the dataset
reader = Reader(line_format='user item rating timestamp', sep=r'::')
train = Dataset.load_from_file('./train.dat', reader=reader)
# build the training set
trainset = train.build_full_trainset()


# # Build the recommender system

# ------------------------------
# ## KNN with Means Algorithm
# define parameters
sim_options = {'name': 'msd',
               'user_based': False,
               'shrinkage': 100}

# build the algorithm
knn = KNNWithMeans(k=40, min_k=1, sim_options=sim_options)
knn.train(trainset)

# read the test file
test = pd.read_csv('./test.csv')

# make predictions
y_hat = np.zeros(len(test))
for i in range(len(test)):
    tmp = test.loc[i]
    uid = str(tmp['user'])
    iid = str(tmp['movie'])
    prediction = knn.predict(uid, iid, verbose=False)
    y_hat[i] = prediction.est

# save prediction
test['rating'] = y_hat
test.to_csv('./mysubmission1.csv', index=False)


# ------------------------------
# ## SVD Algorithm
# build the algorithm
svd = SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0,
          init_std_dev=.1, lr_all=.005, reg_all=.02, lr_bu=None,
          lr_bi=None, lr_pu=None, lr_qi=None, reg_bu=None,
          reg_bi=None, reg_pu=None, reg_qi=None, verbose=False)
svd.train(trainset)

# read the test file
test = pd.read_csv('./test.csv')

# make predictions
y_hat = np.zeros(len(test))
for i in range(len(test)):
    tmp = test.loc[i]
    uid = str(tmp['user'])
    iid = str(tmp['movie'])
    prediction = svd.predict(uid, iid, verbose=False)
    y_hat[i] = prediction.est

# save prediction
test['rating'] = y_hat
test.to_csv('./mysubmission2.csv', index=False)
