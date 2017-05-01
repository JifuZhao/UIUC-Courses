#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# #### Follow the tutorial material from Kaggle (use TF-IDF instead)
# * 1-gram
# * top 5000 words

import warnings
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# # Read data
train = pd.read_csv('./labeledTrainData.tsv', delimiter='\t', quoting=3)

# change the following code based on your desire
test = pd.read_csv('./testData.tsv', delimiter='\t', quoting=3)[:50]

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
                             max_features=5000, ngram_range=(1, 1)) 

# fit and transform the data
all_features = vectorizer.fit_transform(all_review)

# trainsform into array
train_features = all_features[:n_train, :].toarray()
test_features = all_features[n_train:, :].toarray()


# -----------------------------------------------------------------------------
# # Logistic Regression with Lasso

# fit the Logistic model
logit1 = LogisticRegression(penalty='l1', tol=0.0001, C=2.7825549, random_state=2017, 
                            solver='liblinear', n_jobs=-1, verbose=0)
logit1 = logit1.fit(train_features, train_y)

# make predictions
pred1 = logit1.predict_proba(test_features)[:, 1]

# # Logistic Regression with Ridge Penalty

# fit the Logistic model
logit2 = LogisticRegression(penalty='l2', tol=0.0001, C=2.7825549, random_state=2017, 
                            solver='liblinear', n_jobs=-1, verbose=0)
logit2 = logit2.fit(train_features, train_y)

# make predictions
pred2 = logit2.predict_proba(test_features)[:, 1]

# # Make Final Predictions
# merge predictions
pred = 0.5 * pred1 + 0.5 * pred2
label = (pred > 0.5).astype(int)


# -----------------------------------------------------------------------------
# # Define Positive and Negative Words

# find all the words used
words = np.array(vectorizer.get_feature_names())

# get coefficients and importance
logit1coef = logit1.coef_[0, :]
logit2coef = logit2.coef_[0, :]

# make importances relative to max importance
logit1_idx = np.argsort(logit1coef, kind='mergesort')
logit2_idx = np.argsort(logit2coef, kind='mergesort')

# define positive and negative words for eath model
N = 200
logit1_neg = words[logit1_idx][:N]
logit1_pos = words[logit1_idx][-N:]

logit2_neg = words[logit2_idx][:N]
logit2_pos = words[logit2_idx][-N:]

# fine common words in both model
pos_words = list(set(logit1_pos) & set(logit2_pos))
neg_words = list(set(logit1_neg) & set(logit2_neg))


# -----------------------------------------------------------------------------
# # Visualization

# defind the head of the html file
start_string = """
<!DOCTYPE html>
<html>
<head>
<style>@import "textstyle.css"</style>
</head>
<body>
<h1 align="middle">Movie Review Visualization</h1> 
<h3 align="middle"> STAT 542 Project 4 Part II, by Jifu Zhao & Jinsheng Wang </h2>
<h2> 1. Top Positive and Negative Words</h2>
<div>
<img src="./top_words.png" width=40%>
<img src="./importance.png" width=40%>
</div>
<p>
&ensp;&ensp;&ensp;&ensp; The above figures are generated from Logistic Regression 
with Ridge Penalty and Random Forest model
</p><br>
<h2> 2. Detailed Text Analysis</h2>
<h3 align="middle"> <span class="pos">Positive Words</span> vs. <span class="neg">Negative Words</span> </h2>
"""

# process and write to html file
with open('./visualization.html', 'w') as f:
    f.write(start_string)
    f.write('<ul>\n')
    
    for num in range(len(test)):
        idx = test['id'][num][1:-1]
        sentiment = label[num]
        
        # split and replace the review words
        tmp_text = test['review'][num][1:-1]
        tmp_text = tmp_text.replace('\\', '')
        tmp_text = BeautifulSoup(tmp_text, 'lxml').get_text()
        
        # add space after comma that has no space
        tmp_text = re.sub(r'([,]+)(?![ ])', r'\1 ', tmp_text)
        tmp_text = tmp_text.split(' ')
        
        # write the header part of each review
        f.write('<hr>\n&ensp;')
        f.write('<strong> id: </strong> ' + str(idx) + ' &emsp; &emsp;\n')
        f.write('<strong> sentiment: </strong>' + str(sentiment) + ' <br>\n')
        f.write('<hr>\n')
        
        # find all the indexes for the positive and negative words
        pos_idx = []
        neg_idx = []
        for word in pos_words:
            # regular pattern match
#             regex = re.compile('.*(' + word + ').*', re.IGNORECASE)
            regex = re.compile('\\b' + word + '.*', re.IGNORECASE)
            pos_idx += [i for i in range(len(tmp_text)) if regex.search(tmp_text[i])]
        
        for word in neg_words:
            # regular pattern match
#             regex = re.compile('.*(' + word + ').*', re.IGNORECASE)
            regex = re.compile('\\b' + word + '.*', re.IGNORECASE)
            neg_idx += [i for i in range(len(tmp_text)) if regex.search(tmp_text[i])]
            
        # replace all positive words with pre-defined html class
        for i in pos_idx:
            tmp_text[i] = '<span class="pos">' + tmp_text[i] + '</span>'
            
        # replace all negative words with pre-defined html class
        for i in neg_idx:
            tmp_text[i] = '<span class="neg">' + tmp_text[i] + '</span>'
            
        # write into html file
        f.write(' '.join(tmp_text))  
        f.write('\n<br><br>\n\n')
        
    # end of the html file
    f.write('</ul>\n</body>\n</html>\n')

