
# coding: utf-8

# ## Problem 7.2. Chi-squared Test.
# 
# In this problem, you will calculate the chi-squared statstiic and
#   the p-value to accept or reject the hypothesis that the number
#   of flight cancellations depends on the month of the year.

# In[1]:

import numpy as np
import pandas as pd
from scipy import stats


# Suppose you are studying the number of flight cancellations
#   at a small regional airport and observe the following pattern:
#   
#     January           23
#     February          12
#     March             10
#     April             10
#     May               13
#     June               7
#     July              12
#     August            11
#     September         22
#     October            6
#     November           6
#     December          14
# 
# These were the actual 2001 flight cancellations at the Willard airport (CMI), which you should be able to check easily from 2001.csv. It seems like there are a lot of cancellations in January and September, and fewer than average cancellations in June, October, and December.
# 
# In this problem, we will compute the chi-squared statistic to test the hypothesis that the number of flight cancellations depends on the month of the year.
# In other words, we will accept or reject the null hypothesis (the hypothesis that the fluctuations in flight cancellations is entirely due to random fluctuations) using the chi-squared test.
# The chi-squared statistic is defined as
# 
# \begin{equation}
# \chi^2 = \sum_i \frac{\left(O_i - E_i\right)^2}{E_i}.
# \end{equation}
# 
# We will also compute the corresponding p-value of this distribution.
# For example, if the p-value is, say, 50%, then it means that there's a 50% chance and that the apparent pattern is due to chance. On the other hand, if the p-value is something small like 0.1%, then this pattern happens only one time in 1000 when the number of cancellations does not depend on the month of the year.  
# In this problem, we will compute the $\chi^2$ statistic defined as
# 
# 
# 
# and the corresponding $p$-value of this distribution.
# 
# I summarized the number of flight cancellations in the following array:

# In[2]:

cancelled = np.array([23, 12, 10, 10, 13,  7, 12, 11, 22,  6,  6, 14])


# I will break down each step in separate functions to make sure you know
#   how to calculate $\chi^2$. But note that you would normally combine
#   all the steps in a single function.
#   Also note that there's a SciPy function
#   that does this for you:
#   [`scipy.stats.chisquare()`](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.chisquare.html).
#   You may **not** use this function to write your code, but you can use it to check your answer.
#   
# The $\chi^2$ test is covered in Section 7.7 of *Think Stats*.
# 
# First, we define the cells to be the twelve months of the year.
# 
# ### get\_expected()
# 
# Second, write a function named `get_expected()` that takes a Numpy array
#   and returns a Numpy array with the number of flight cancellations
#   expected under the *null hypothesis*.
#   Under the null hypothesis, we expect a uniform distribution,
#   i.e. the same number of flight cancellations in each month.
#   However, we also want our function to handle different types of data,
#   so do not simply add up all the numbers and divide by 12.
#   Instead, you should use the length of the array.
#   For example, if the input array is `np.array([1.0, 2.0, 3.0, 4.0, 5.0]`, the `get_expected()`
#   should return `np.array([3.0, 3.0, 3.0, 3.0, 3.0])`.

# In[3]:

def get_expected(x):
    '''
    Takes a Numpy array and returns a Numpy array of the same length
    that represents the expected values (uniform distribution).
    
    Parameters
    ----------
    x: A Numpy array.
    
    Returns
    -------
    A Numpy array of the same length as x.
    
    Example
    -------
    >>> import numpy as np
    >>> x = get_expected([1., 2.])
    >>> x = get_expected(np.array([2., 1., 3., 4.]))
    >>> x
    array([ 2.5,  2.5,  2.5,  2.5])
    '''
    
    # your code goes here
    
    N = len(x)
    expected = np.array([x.sum()/N]*N)
    
    return expected


# ### get\_diff()
# 
# Next, write a function named `get_diff()` that takes two Numpy arrays,
#    an array of the observed values and an array of the expected values.
#    For each cell, the funtion calculates the difference between the observed value $O_i$ and the expected value $E_i$.
#    Thus, it should return a Numpy array of $O_i$ - $E_i$.

# In[4]:

def get_diff(observed, expected):
    '''
    Takes two Numpy arrays, the observed values and the expected values,
    and returns a Numpy array of the differences.
    
    Parameters
    ----------
    observed: A Numpy array.
    expected: A Numpy array.
    
    Returns
    -------
    A Numpy array.
    
    Example
    -------
    >>> import numpy as np
    >>> x = np.array([2., 4., 6.])
    >>> y = np.array([3., 6., 9.])
    >>> z = get_diff(x, y)
    >>> z
    array([-1., -2., -3.])
    '''
    
    # your code goes here
    
    diff = observed - expected

    return diff


# ### get_chi2()
# 
# Next, write a function named `get_chi2()` that takes two Numpy arrays,
#   an array of the difference between the observed values and the expected values
#   and an array of the expected values.
#   It should return a single float that represents the $\chi^2$ statistic.

# In[5]:

def get_chi2(diff, expected):
    '''
    Takes O - E and computes chi-squared.
    
    Parameters
    ----------
    diff: A Numpy array. O - E
    expected: A numpy array. E
    
    Returns
    -------
    A float. The chi-squared value.
    
    Example
    -------
    >>> import numpy as np
    >>> x = np.array([1., 2., 3.])
    >>> y = np.array([1., 2., 3.])
    >>> z = get_chi2(x, y)
    >>> z
    '''
    
    # your code goes here
    
    chi2 = (diff**2 / expected).sum()
    
    return chi2


# ### get\_p\_value()
# 
# Finally, write a function named `get_p_value()` that takes two arguments,
#   the $\chi^2$ value (a float) and a degree of freedom (an int).
#   The $p$-value can easily be calculated by using
#   [scipy.stats.chi2.cdf()](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html).
#   The $p$-value is given simply by 1 minus the CDF of the $\chi^2$ distribution
#   (See [Chi-squared distribution](http://en.wikipedia.org/wiki/Chi-squared_distribution)).
#   When using `scipy.stats.chi2.cdf()`, you should specify the `df` option,
#   which corresponds to the degrees of freedom.
#   The [degrees of freedom](http://en.wikipedia.org/wiki/Degrees_of_freedom_%28statistics%29)
#   is the number of cells minus 1, which in our case is 11
#   (the twelve months of the year minus 1).

# In[6]:

def get_p_value(chi2, dof):
    '''
    Takes chi-squared and degrees of freedom
    and computes the p-value.
    
    Parameters
    ----------
    chi2: A float.
    dof: An int. Degrees of freedom.
    
    Returns
    -------
    >>> import numpy as np
    >>> x = 1.0
    >>> y = 2
    >>> z = get_p_value(x, y)
    0.60653065971263342
    '''
    
    # your code goes here
    
    p =1 - stats.chi2.cdf(chi2, dof)
    
    return p


# When you run the following cell, you should get
# 
#     The chi square value is 27.3 and the p-value is 4.2e-03.
#     
# That means there's a 0.4% chance that the observed pattern is due to chance.
# I think we can safely reject the null hypothesis.

# In[7]:

expected = get_expected(cancelled)
diff = get_diff(cancelled, expected)
chi2 = get_chi2(diff, expected)
p = get_p_value(chi2, len(expected) - 1)
print('The chi square value is {0:.1f} and the p-value is {1:.1e}.'.format(chi2, p))


# In[ ]:



