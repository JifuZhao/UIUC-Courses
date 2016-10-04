
# coding: utf-8

# ## Problem 6.3. PMF and CDF.
# 
# In this problem, you will compute and plot the
#   cumulative distribution function (CDF) of arrival delay.

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# First, write a function named `get_cdf()` that takes an array
#   and returns a tuple that represents the $x$ and $y$ axes of the (empirical) CDF.
#   
# Here is a very easy algorithm to implement this function.
#   See the definition of
#   [empirical distribution function](http://en.wikipedia.org/wiki/Empirical_distribution_function)
#   on Wikipedia. That means we can do the following to produce
#   the empirical CDF:
# 
# 1. Use Pandas to read the specified column.
# 2. Pandas will automatically replace missing values `'NA'` with
#    `numpy.nan` (Not A Number).
#    Use [`numpy.isfinite()`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.isfinite.html)
#    to mask out these missing values.
# 3. Use [`numpy.sort()`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html)
#    to sort the array (with no missing values) in ascending order.
#    This will be our $x$-axis.
# 4. Create an array of $\frac{1}{N}$, $\frac{2}{N}$, ..., $1$,
#    where $N$ is the length of the sorted array (the $x$-axis).
#    This will be our $y$-axis.
#    All you have to do is use `np.arange()` to make an array
#    of length $N$,
#    and divide each element by $N$.
# 
# According to Wikipedia, the resulting empirical CDF is an unbiased estimator for the true CDF.
# 
# Note: Do NOT use numpy.histogram() function to create a CDF.
#   It uses binning, which might be useful in other cases but not in this case.
#   The method I outlined above is a better characterization of the true CDF.

# In[2]:

#%%writefile FirstName_LastName_cdf.py

def get_cdf(filename, column):
    '''
    Reads a specific column of airline on-time performance CSV file,
    and returns a tuple of arrays that represent the x and y axes of
    cumulative distribution function.
    
    Parameters
    ----------
    filename(str): The filename of a CSV file.
    column(str): The column header.
    
    Returns
    -------
    A tuple of two numpy arrays of equal length.
    The first array represents the x axis of CDF.
    The second array represents the y axis of CDF.
    '''
    
    # your code goes here
    df = pd.read_csv(filename, encoding='latin1',usecols = [column], na_values= None)
    df = df.dropna(axis = 0, how = 'any')
    cdf_x = np.sort(df[column], 0)
    N = len(cdf_x) 
    cdf_y = np.arange(1,N+1,1)/N

    return cdf_x, cdf_y


# The `%%writefile` magic will create a file named `FirstName_LastName_cdf.py`.
# Rename and submit this file along with your `.ipynb` file.
# 
# Next, use the `get_cdf()` function to create a CDF plot.
# Your plot should show both the empirical CDF calculated
# from `ArrDelay` column of `2001.csv` file
# and the CDF of a Guassian distribution.
# You can obtain the CDF of a Guassian distribution by using
# [`scipy.stats.norm.cdf()`](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html).
# Use `loc=-3.5` and `scale=10.7`
# (which were obtained by fitting a Gaussian curve
# on the PDF).

# In[6]:

# your code goes here
# This process may take some time to get the final result

filename = '/data/airline/2001.csv'
column = 'ArrDelay'
x,y1 = get_cdf(filename, column)
y2 = stats.norm.cdf(x, loc=-3.5,scale=10.7)

fig, ax = plt.subplots()
ax.plot(x, y1, 'r',label = 'data')
ax.plot(x, y2, label = 'model')

ax.set_xlabel('Arrival Delay (minutes)')
ax.set_ylabel('Probability')
ax.set_title('Arrival Delay CDF (by Jifu Zhao)')
ax.set_xlim(-100,100)
plt.legend(loc = 'upper left')

plt.show()


# In[ ]:



