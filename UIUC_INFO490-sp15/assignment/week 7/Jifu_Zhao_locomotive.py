
# coding: utf-8

# ## Problem 7.1. The Locomotive Problem
# 
# In this problem, you will estimate the number of aircrafts in the aircrafts in the U.S. by using Bayesian statistics.

# In[1]:

import numpy as np
import pandas as pd
import re


# The locomotive problem is found in Section 3.2 of *Think Bayes* by Allen B. Downey.
#   We will rephrase the locomotive problem and apply it to our 2001 flights data:
# 
# > The tail number of aircrafts in the 2001 flights data are labeled using the serial numbers 1, 2, ..., $n$.
# > You see an airplane with the tail number N363. Estimate how many aircrafts there are in the U.S.
# 
# All civil aircrafts in the U.S. have tail numbers that begin with the letter N.
#   (source: [Wikipedia](http://en.wikipedia.org/wiki/Aircraft_registration))
#   The serial numbers actually are more complicated than 1, 2, ...
#   There could also be letters at the end of a tail number,
#   but let's simplify the problem and assume that the tail numbers are nicely ordered
#   from 1 to $n$.
#   
# Where did the tail number N363 come from? I picked a random row in the 2001 flights data set.

# In[2]:

# change the path to the location of 2001.csv on your machine.
fpath = '/data/airline/2001.csv'
df = pd.read_csv(fpath, encoding='latin-1', usecols=('TailNum',))
np.random.seed(490) # this number ensures that you get the same number from random number generator
rand_row = np.random.randint(0, high=len(df))
tail_num = df.ix[rand_row].TailNum
print(tail_num)


# It seems like encoding is all messed up. Remember regular expressions (regex)?
#   Let's use regex to extract the digits.

# In[3]:

tail_num_num = int(re.sub('\D', '', tail_num))
print(tail_num_num)


# In this problem, we will completely rewrite Downey's code by eliminating the class structure and using Numpy, because
# 
# 1. Classes have their advantages, but they make it hard to see what is really going on under the hood, and our goal in this problem is to understand the underlying concepts,
# 2. Using Numpy means no *for* loops, which in my opinion makes it easier to understand what's going on, and
# 3. Downey mostly uses pure Python. This is very slow, so we will use Numpy.
# 
# Your task in this problem is to write the following five functions:
# 
# - hypotheses()
# - uniform_prior()
# - likelihood()
# - posterior()
# - estimate()
# 
# Note that if you use Numpy functions correctly, each function will not take more than one or two lines to write. I will give you hints on how to do this, but you don't have to follow my recommendations as long as your functions do what they are supposed to do, i.e. take the specified parameters and return the correct Numpy arrays. But remember that all functions must return Numpy arrays, so if you use *for* loops you code will be very inefficient.
# 
# #### Function: hypotheses()
# 
# - Write a function named `hypotheses()` that takes two integers (the first tail number 1 and the final tail number $n$) and returns a Numpy array of integers [1, 2, ..., $n$].
# 
# Note that the number $n$ will be used for building a uniform prior.
#   So what is our prior belief regarding the number of aircrafts before seeing the data?
#   Not much, so let's a very simple prior distribution for this problem,
#   and assume that $n$ is equally likely to be anywhere from 1 to 10000.
# 
# Hint: Use [numpy.arange()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).

# In[5]:

def hypotheses(start=1, end=10001):
    '''
    Takes two integers (the first serial number 1 and the final serial number n + 1).
    Returns a Numpy array of integers [1, 2, ..., n].
    
    Parameters
    ----------
    start(int): The start of the array. Optional.
    end(int): The end of the arary. Optional.
    
    Returns
    -------
    A Numpy array of integers [1, 2, ..., n]
    
    Examples
    --------
    >>> hypotheses(1, 5)
    array([1, 2, 3, 4])
    '''

    # your code goes here
    
    result = np.arange(start, end, 1 )
    
    return result


# #### Function: uniform_prior()
# 
# - Write a function named `uniform_prior()` that takes a Numpy array (an array of hypotheses) and returns a Numpy array of floats (an array of uniform prior).
# 
# The prior is $P(H)$ in Bayes' theorem:
# 
# $P\left(H\mid D\right) = \frac{P\left(H\right)P\left(D\mid H\right)}{P\left(D\right)}$
# 
# Here, we use a uniform prior, i.e. all elements in the array have the same values, $\frac{1}{N}$.
# 
# Hint: See [`numpy.ones_like()`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ones_like.html). Create an array of same length as hypotheses with all ones. (If the input array to `ones_like()` is an integer array, the output array will also be an integer array; you might want to specify *dtype* to make it a floating point array.) Divide all elements by the length of hypotheses.

# In[7]:

def uniform_prior(hypotheses):
    '''
    Takes an array (an array of hypotheses) and
    returns a Numpy array of floats (an array of uniform prior).
    
    Parameters
    ----------
    hypotheses: A numpy array of length n.
    
    Returns
    -------
    A numpy of length n.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> uniform_prior(x)
    array([ 0.2,  0.2,  0.2,  0.2,  0.2])
    '''
    
    # your code goes here
    
    N = len(hypotheses)
    uni = np.ones_like(hypotheses, dtype=np.float)
    result = uni/N
    
    return result


# #### Function: likelihood()
# 
# - Write a function named `likelihood()` that takes an integer (data) and a Numpy array (hypotheses), and returns a Numpy array of floats (likelihood).
# 
# The likelihood is $P\left(D\mid H\right)$ in Bayes' theorem. Recall that hypotheses is a Numpy array of [1, 2, ..., $n$]. If the $n$-th element of hypotheses is smaller than data, the $n$-th element of likelihood is zero because the serial number cannot be greater than the number of households. On the other hand, if the $n$-th element of hypotheses is greater than or equal to data, the question becomes what is the chance of getting a particular serial number given that there are `hypotheses[n]` persons? Thus, if `hypotheses[n] >= data`, `likelihood[n]` is `1.0 / hypotheses[n]`, and `0` otherwise.
# 
# Hint: Say you have the following Numpy arrays:

# In[61]:

x = np.array([1.0, 2.0, 3.0])
y = np.array([0.5, 2.5, 2.5])


# You want to compare them *element-wise* and create a Boolean array of `True` if `x >= y` and `False` if `x < y`. An easy way to do this is

# In[62]:

(x > y)


# See also [`numpy.ndarray.astype()`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html). That means we can convert our Boolean array to, say, an array of floats with

# In[63]:

(x > y).astype(np.float)


# Write the `likelihood()` function below.

# In[8]:

def likelihood(data, hypotheses):
    '''
    Takes an integer (data) and an array (hypotheses)
    Returns a Numpy array of floats (likelihood).
    
    Parameters
    ----------
    data: An int.
    hypotheses: A numpy array.
    
    Returns
    -------
    A Numpy array.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> likelihood(1, x)
    array([ 1.        ,  0.5       ,  0.33333333,  0.25      ,  0.2       ])
    >>> likelihood(2, x)
    array([ 0.        ,  0.5       ,  0.33333333,  0.25      ,  0.2       ])
    >>> likelihood(3, x)
    array([ 0.        ,  0.        ,  0.33333333,  0.25      ,  0.2       ])
    >>> likelihood(6, x)
    array([ 0.,  0.,  0.,  0.,  0.])
    '''
    
    # your code goes here
    
    compare = (hypotheses >= data).astype(np.float)
    result = compare/hypotheses
    
    return result


# #### Function: posterior()
# 
# - Write a function named `get_posterior()` that takes an integer (data) and a Numpy array (hypotheses), and returns a Numpy array of floats (posterior).
# 
# From Bayes' theorem, the posterior $P(D)$ is equal to prior times likelihood (divided by the normalizing constant). Thus, the `get_posterior()` function should
# 
# 1. Call `get_uniform_prior()` with `hypotheses` as a parameter,
# 2. Call `get_likelihood()` using `data` and `hypotheses`,
# 3. Multiply two arrays element-wise,
# 4. And normalize to produce a posterior.
# 
# Hint: A simple multiplication of two Numpy arrays performs the multiplication *element-wise*. For example,

# In[66]:

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
x * y


# This element-wise multiplication is exactly what we want; we want to multiply the prior and the likelihood element-wise.
# 
# Hint: Don't forget to normalize. Since adding up all elements of the posterior array should give you 1.0,
#   an easy way to normalize is to divide each element by the sum of all elements.
#   See [`numpy.sum()`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html).
#   For example, 

# In[67]:

x = np.array([1.0, 2.0, 3.0, 4.0])
x / x.sum()


# Write the `posteior()` function below.

# In[9]:

def posterior(data, hypotheses):
    '''
    Takes an integer (data) and an array (hypotheses).
    Returns a Numpy array of floats (posterior).
    
    Parameters
    ----------
    data: An int.
    hypoteses: A Numpy array.
    
    Returns
    -------
    A Numpy array.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> posterior(1, x)
    array([ 0.4379562 ,  0.2189781 ,  0.1459854 ,  0.10948905,  0.08759124])
    >>> posterior(2, x)
    array([ 0.        ,  0.38961039,  0.25974026,  0.19480519,  0.15584416])
    >>> osterior(3, x)
    array([ 0.        ,  0.        ,  0.42553191,  0.31914894,  0.25531915])
    >>> posterior(5, x)
    array([ 0.,  0.,  0.,  0.,  1.])
    '''

    # your code goes here
    
    prior = uniform_prior(hypotheses)
    like = likelihood(data, hypotheses)
    post = prior*like
    result = post/post.sum()
    
    return result


# #### Function: estimate()
# 
# - Write a function named `estimate()` that takes two Numpy arrays (posterior and hypotheses) and returns a floating point (estimate).
# 
# Hint: Again, you can use element-wise multiplication of `hypotheses` and `posterior`, and our estimate is the sum of all elements in the product array.

# In[10]:

def estimate(posterior, hypotheses):
    '''
    Takes a two arrays (posterior and hypotheses).
    Returns a float (estimate).
    
    Parameters
    ----------
    posterior: A Numpy array.
    hypotheses: A Numpy array.
    
    Returns
    -------
    A float.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.25, 0.25, 0.5])
    >>> y = np.array([1, 2, 3])
    >>> estimate(x, y)
    2.25
    '''

    # your code goes here
    
    result = float((posterior * hypotheses).sum())
    
    return result


# When you run the following cell, you should get
# 
#     The number of aircrafts in the U.S. is 2905.

# In[11]:

hypo = hypotheses()
like = likelihood(tail_num_num, hypo)
post = posterior(tail_num_num, hypo)
estimate = estimate(post, hypo)
print('The number of aircrafts in the U.S. is {0:.0f}.'.format(estimate))


# Note that our method significantly underestimates the number of aircrafts.
#   This is due to a number of factors: we assumed that the tail numbers are determined in a simple manner,
#   and also assumed a uniform prior distribution.
#   In a more sophisticated model, we would have to account for these factors to produce
#   a more accurate esimate.

# In[ ]:



