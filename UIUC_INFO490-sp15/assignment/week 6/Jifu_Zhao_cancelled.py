
# coding: utf-8

# ## Problem 6.2. Flight Cancellations by Month.
# 
# In this problem, you will use Panda's
#   [`groupby()`](http://pandas.pydata.org/pandas-docs/stable/groupby.html)
#   and [`aggregate()`](http://pandas.pydata.org/pandas-docs/stable/groupby.html)
#   functions to compute and plot the number of flight cancellations
#   in each month of 2001.

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# First, write a function named `get_month_cancelled()` that takes a filename (str)
#   and returns a `pd.DataFrame` that is indexed by the **names** of the months
#   and has only one column `Cancelled`, the number of flight cancellations in each month.
# 
# - Don't forget to set the `encoding` option.
# - Again, if you try to read in all 29 columns, your code will be very slow.
#   Use `usecols` to read only two columns, `Month` and `Cancelled`.
# - If you don't set the indices, they will be just numbers, e.g. 0, 1, 2...
#   Use the following list to set the indices.
#   Copy/paste (rather than type) since even a single typo will cause problems for autograding.
#   
#       ['January', 'February', 'March', 'April', 'May', 'June',
#        'July', 'August', 'September', 'October', 'November', 'December']
#    
# - When you call `get_month_cancelled('2001.csv'), you should get the following DataFrame.
# 
#                    Cancelled
#         January        19891
#         February       17448
#         March          17876
#         April          11414
#         May             9452
#         June           15509
#         July           11286
#         August         13318
#         September      99324
#         October         6850
#         November        4497
#         December        4333
# 
#         [12 rows x 1 columns]
#         
# - The `%%writefile` magic writes the `get_month_cancelled()` function
#   to a file named `FirstName_LastName_cancelled.py`.
#   Edit the command or rename the file, and upload this file along
#   with your `.ipynb` file.

# In[2]:

#%%writefile FirstName_LastName_cancelled.py

def get_month_cancelled(filename):
    '''
    Reads the "Month" and "Cancelled" columns of a CSV file
    and returns a Pandas DataFrame with only one column "Cancelled"
    indexed by the months.
    
    Parameters
    ----------
    filename(str): The filename of the CSV file.
    
    Returns
    -------
    pandas.DataFrame: "Cancelled" column, indexed by names of the months
    '''
    
    # your code goes here
    
    df = pd.read_csv(filename, encoding='latin1',usecols = ['Month','Cancelled']).dropna(axis=0, how = 'any')
    month_cancelled = df.groupby('Month').aggregate(sum)
    month_cancelled.index = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    return month_cancelled


# When you run the following cell, you should get
# 
#                Cancelled
#     January        19891
#     February       17448
#     March          17876
#     April          11414
#     May             9452
#     June           15509
#     July           11286
#     August         13318
#     September      99324
#     October         6850
#     November        4497
#     December        4333
# 
#     [12 rows x 1 columns]

# In[3]:

month_cancelled = get_month_cancelled('/data/airline/2001.csv')
print(month_cancelled)


# Run the following cell to plot a bar histogram.

# In[4]:

month_cancelled.plot(kind='bar')


# In[9]:

get_ipython().run_cell_magic('writefile(Jifu_Zhao_cancelled.py)', '', '')


# In[ ]:

get_ipython().run_cell_magic('writefile', '', '')

