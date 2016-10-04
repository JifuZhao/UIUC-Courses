
# coding: utf-8

# ## Problem 4.2. Simple Statistics II.
# 
# This problem is a continuation of problem 4.1. Recall that you wrote a function named `get_stats()` that takes a list and returns a tuple of minimum, maximum, mean, and median. To use this function, you have to convert your IPython notebook to a regular `.py` file. One way to do this is to use the IPython `%%script%%` magic function; open up a new notebook, and in an IPython notebook cell, type (assuming the filename of your IPython notebook from Problem 4.1 is `stats.ipynb`):
# 
#     %%bash
#     ipython3 nbconvert --to python stats.ipynb
# 
# and press <kbd>shift</kbd> + <kbd>enter</kbd>. This will create a Python script
# named `stats.py`. We will import this as a module in `stats2.ipynb`:

# In[1]:

from stats import get_stats


# We will use the function `get_stats()` to compute basic statistics of a number of columns from the arline performance dataset we downloaded in week 2. Namely, we will use the following columns:
# 
# - Column 15, "ArrDelay": arrival delay, in minutes,
# - Column 16, "DepDelay": departure delay, in minutes, and
# - Column 19, "Distance": distance, in miles.
# 
# To extract these columns from the CSV file,
# 
# - Write a function named `get_column(filename, n, header = True)` that reads the `n`-th column from a file and returns a list of integers.
# 
# - You may assume that the column is made of integers.
# 
# - We will also use the optional argument `header` because the first line of our file lists the names of the columns, but we might want to turn this off to handle a file that doesn't have a header.
# 
# - Use a combination of `with` statement and `open()` function to open `filename` in the `get_column()` function.
#   
#   Tip: When I tried to use `open()` to read `2001.csv`, I had the following error:
#   
#         'utf-8' codec can't decode byte 0xe4 in position 343: invalid continuation byte
#         
#   You can avoid this error by using `encoding='latin-1'` option in `open()`.
#   
# - Skip the first line if the `header` parameter is `True`; do not skip if it's `False`.
# 
# - Some columns have missing values `'NA'`, and you need a way to handle these
#   missing values. If the `n`-th column is missing, you should **not** include
#   that column in `result`; that is, skip all rows with `'NA'`.
#   As a result, lists returned from different
#   columns may have different lengths.

# In[2]:

def get_column(filename, n, header=True):
    '''
    Returns a list from reading the specified column in the CSV file.

    Parameters
    __________
    filename(str): Input file name in Comma Separated Values (CSV) format
    n(int): Column number. The first column starts at 0. The column must be
            a list of integers.
    header(bool): If True, the first line of file is column names.
                  Default: True.

    Examples
    ________
    >>> get_column('/data/airline/2001.csv', 14)[:10]
    [-3, 4, 23, 10, 20, -3, -10, -12, -9, -1]
    >>> get_column('/data/airline/2001.csv', 15)[-10:]
    [-4, -5, -8, 4, -7, 4, 8, -4, -4, 9]
    '''
    result = []
    
    # your code goes here
    
    with open(filename, encoding='latin-1') as a_file:           
        for a_line in a_file:
            tmp = a_line.split(',')
            if (tmp[n] == 'NA'):
                continue
            result.append(tmp[n])
        
        if (header == True):
            del result[0]
            
        for m in range(len(result)):
            result[m] = int(result[m])
    
    return result


# We also want to print out the results in a nicely formatted manner.
# 
# - The `print_stats(input_list, title=None)` function is already written for you.
#   You don't need to write this function.
# 
# It takes a list of integers and prints out the basic statistics.

# In[3]:

def print_stats(input_list, title=None):
    '''
    Computes minimum, maximum, mean, and median using get_stats function from
      stats module, and prints them out in a nice format.

    Parameters:
      input_list(list): a list representing a column
      title(str): Optional. If given, title is printed out before the stats.

    Examples:
    >>> print_stats(list(range(50)))
    Minimum: 0
    Maximum: 49
    Mean: 24.5
    Median: 24.5
    >>> print_stats(list(range(100)), title = 'Stats!')
    Stats!
    Minimum: 0
    Maximum: 99
    Mean: 49.5
    Median: 49.5
    '''
    if title is not None:
        print(title)
        
    minimum, maximum, mean, median = get_stats(input_list)
    print('Minimum: {0}\n'
          'Maximum: {1}\n'
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

# In[4]:

#warning: this could take a while.
filename = '/data/airline/2001.csv' # 2001 airline on-time performance dataset

arr_delay = get_column(filename, 14)
print_stats(arr_delay, "Arrival delay, in minutes.")


# When you run the following cell, you should get
# 
#     Departure delay, in minutes.
#     Minimum: -204
#     Maximum: 1692
#     Mean: 8.15
#     Median: 0.00

# In[5]:

#warning: this could take a while.
dep_delay = get_column(filename, 15)
print_stats(dep_delay, "Departure delay, in minutes.")


# When you run the following cell, you should get
# 
#     Distance, in miles.
#     Minimum: 21
#     Maximum: 4962
#     Mean: 733.03
#     Median: 571.00

# In[6]:

#warning: this could take a while.
distance = get_column(filename, 18)
print_stats(distance, "Distance, in miles.")


# In[ ]:



