#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# #### Follow the tutorial material from Kaggle (use TF-IDF instead)
# * 2-gram
# * top 10000 words

import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# # Read data
train = pd.read_csv('./labeledTrainData.tsv', delimiter='\t', quoting=3)
test = pd.read_csv('./testData.tsv', delimiter='\t', quoting=3)

# get the number of training and test examples
n_train = len(train)
n_test = len(test)


# -----------------------------------------------------------------------------
# # Data Cleaning and Processing
def review2words(review):
    """ function to convert input review into string of words """
    # Remove HTML
    review_text = BeautifulSoup(review, 'lxml').get_text() 

    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             

    # Join the words and return the result.
    return " ".join(words)


# get train label
train_y = train['sentiment'].values

# transform reviews into words list
train_review = list(map(review2words, train['review']))
test_review = list(map(review2words, test['review']))

# combine train and test reviews
all_review = train_review + test_review

# perform TF-IDF transformation
vectorizer = TfidfVectorizer(min_df=3, analyzer="word", strip_accents='unicode', 
                             sublinear_tf=True, stop_words='english', 
                             max_features=10000, ngram_range=(1, 2)) 

# fit and transform the data
all_features = vectorizer.fit_transform(all_review)

# trainsform into array
train_features = all_features[:n_train, :].toarray()
test_features = all_features[n_train:, :].toarray()


# -----------------------------------------------------------------------------
# # Logistic Regression with Ridge Penalty

# fit the Logistic model
logit2 = LogisticRegression(penalty='l2', tol=0.0001, C=2.7825549, random_state=2017, 
                            solver='liblinear', n_jobs=-1, verbose=0)
logit2 = logit2.fit(train_features, train_y)

# make predictions
train_pred1 = logit2.predict_proba(train_features)[:, 1]
test_pred1 = logit2.predict_proba(test_features)[:, 1]


# -----------------------------------------------------------------------------
# # Logistic Regression with Lasso Penalty

# fit the Logistic model
logit1 = LogisticRegression(penalty='l1', tol=0.0001, C=2.7825549, random_state=2017, 
                            solver='liblinear', n_jobs=-1, verbose=0)
logit1 = logit1.fit(train_features, train_y)

# make predictions
train_pred2 = logit1.predict_proba(train_features)[:, 1]
test_pred2 = logit1.predict_proba(test_features)[:, 1]


# -----------------------------------------------------------------------------
# # Multinomial Naive Bayes Model

# build the NB model
nb = MultinomialNB(alpha=5.0, fit_prior=True, class_prior=None)
nb = nb.fit(train_features, train_y)

# make predictions
train_pred3 = nb.predict_proba(train_features)[:, 1]
test_pred3 = nb.predict_proba(test_features)[:, 1]


# -----------------------------------------------------------------------------
# # AdaBoost

# build the AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=400, learning_rate=1.0, 
                              algorithm='SAMME.R', random_state=2017)
adaboost = adaboost.fit(train_features, train_y)

# make predictions
train_pred4 = adaboost.predict_proba(train_features)[:, 1]
test_pred4 = adaboost.predict_proba(test_features)[:, 1]


# -----------------------------------------------------------------------------
# # Gradient Boosting 

# build the Gradient Boosting classifier
gbm = GradientBoostingClassifier(learning_rate=0.2, n_estimators=500, subsample=1.0,
                                 max_features='auto', min_samples_split=2, random_state=2017, 
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3)
gbm = gbm.fit(train_features, train_y)

# make predictions
train_pred5 = gbm.predict_proba(train_features)[:, 1]
test_pred5 = gbm.predict_proba(test_features)[:, 1]


# -----------------------------------------------------------------------------
# # Random Forest Model

# Initialize a Random Forest classifier
forest = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2017, 
                                oob_score=True, max_features='auto') 

# Fit the forest to the training set
forest = forest.fit(train_features, train_y)

# make predictions
train_pred6 = forest.predict_proba(train_features)[:, 1]
test_pred6 = forest.predict_proba(test_features)[:, 1]


# -----------------------------------------------------------------------------
# # Merge Score

# define weights
w = np.array([2, 2, 1, 1, 1, 1]) / 8.0

# weighted train and test prediction
train_pred = w[0] * train_pred1 + w[1] * train_pred2 + w[2] * train_pred3 + \
    w[3] * train_pred4 + w[4] * train_pred5 + w[5] * train_pred6
    
test_pred = w[0] * test_pred1 + w[1] * test_pred2 + w[2] * test_pred3 + \
    w[3] * test_pred4 + w[4] * test_pred5 + w[5] * test_pred6
    
# save prediction into local files
output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
output.to_csv("./mysubmission.csv", index=False, quoting=3)

