
# coding: utf-8

# ## Problem 6.1. Simple Statistics Using Pandas.
# 
# In the previous weeks, we have seen different ways to read selected columns from the census CSV file
#   and calculate basic statistics. In this problem, we will see how easy it is to perform the same task
#   using Pandas. In particular, we will rewrite `get_stats()` function from
#   [Problem 4.1](https://github.com/INFO490/spring2015/blob/master/week04/p1.md)
#   and `get_column()` function from
#   [Problem 4.2](https://github.com/INFO490/spring2015/blob/master/week04/p2.md).
#   Remember, the purpose of this problem is to let you experience how easy it is to make
#   a data table using Pandas. Don't overthink it.

# In[1]:

import pandas as pd


# First, write a function named `get_column()` that takes a filename (string) and a column name (string),
#   and returns a `pandas.DataFrame`. Remember that `encoding='latin-1'`.
# 
# Another useful tip: if you try
#   to read the entire file, it will take a long time. Read in only one column by specifying the column
#   you wish to read with the
#   [`usecols`](http://pandas.pydata.org/pandas-docs/stable/io.html#io-read-csv-table) option.
#   Therefore, the `get_column` function should return a DataFrame with only **one** column.
# 
# With Pandas, the `get_column()` function can be written in one line.

# In[2]:

def get_column(filename, column):
    '''
    Reads the specified column of airline on-time performance CSV file,
    which is in 'latin-1' encoding.
    Returns a Pandas DataFrame with only one column.
    
    Parameters
    ----------
    filename(str): The file name.
    column(str): The column header.
    
    Returns
    -------
    A pandas.DataFrame object that has only column.
    
    Examples
    --------
    arr_delay = get_column('/data/airline/2001.csv', 'ArrDelay')
    '''
    
    # your code goes here
    df = pd.read_csv(filename, encoding='latin1',usecols = [column]).dropna(axis=0, how = 'any')
    
    return df


# Next, write a function named `get_stats()` that takes a `pandas.DataFrame` and a column name (string),
#   and return the minimum, maximum, mean, and median (all floats) of the column.

# In[3]:

def get_stats(df, column):
    '''
    Calculates the mininum, maximum, mean, and median values
    of a column from a Pandas DataFrame object.
    
    Parameters
    ----------
    df(pandas.DataFrame): A Pandas DataFrame.
    column(str): The column header.
    
    Returns
    -------
    minimum(float)
    maximum(float)
    mean(float)
    median(float)
    '''
    
    # your code goes here

    minimum = float(min(df[column]))
    maximum = float(max(df[column]))
    mean = float(df[column].mean())
    median = float(df[column].median())
    
    return minimum, maximum, mean, median


# We will use the same function from
#   [Problem 4.1](https://github.com/INFO490/spring2015/blob/master/week04/p1.md)
#   to print out the statistics in a nicley formatted manner.

# In[4]:

def print_stats(df, column, title=None):
    '''
    Computes minimum, maximum, mean, and median using get_stats function from
      pdstats module, and prints them out in a nice format.

    Parameters:
      df(pandas.DataFrame): a Pandas DataFrame
      column(str): The column header.
      title(str): Optional. If given, title is printed out before the stats.
    '''
    if title is not None:
        print(title)
        
    minimum, maximum, mean, median = get_stats(df, column)
    print('Minimum: {0:.0f}\n'
          'Maximum: {1:.0f}\n'
          'Mean: {2:.2f}\n'
          'Median: {3:.2f}'.format(minimum, maximum, mean, median))
    return None


# When you run the following cell, you should get
# 
#     Arrival delay, in minutes.
#     Minimum: -1116
#     Maximum: 1688
#     Mean: 5.53
#     Median: -2.00

# In[5]:

arr_delay = get_column('/data/airline/2001.csv', 'ArrDelay')
print_stats(arr_delay, 'ArrDelay', 'Arrival delay, in minutes.')


# When you run the following cell, you should get
# 
#     Departure delay, in minutes.
#     Minimum: -204
#     Maximum: 1692
#     Mean: 8.15
#     Median: 0.00

# In[6]:

dep_delay = get_column('/data/airline/2001.csv', 'DepDelay')
print_stats(dep_delay, 'DepDelay', 'Departure delay, in minutes.')


# When you run the following cell, you should get
# 
#     Distance, in miles.
#     Minimum: 21
#     Maximum: 4962
#     Mean: 733.03
#     Median: 571.00

# In[7]:

distance = get_column('/data/airline/2001.csv', 'Distance')
print_stats(distance, 'Distance', 'Distance, in miles.')


# When you are done, run the following cell, which produces `pdstats.py`.
# Rename and submit this `.py` file along with your `.ipynb` file.

# In[8]:

get_ipython().run_cell_magic('bash', '', 'ipython3 nbconvert --to python pdstats.ipynb')


# In[ ]:



