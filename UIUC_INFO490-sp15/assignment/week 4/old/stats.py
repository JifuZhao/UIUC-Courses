
# coding: utf-8

# ## Problem 4.1. Simple Statistics I.
# 
# In this problem, you will write a function that calculates the minimum, maximum, mean, and median values from a list of integers.
# 
#  - Write a function named `get_stats()` takes one argument, which should be a list of integers, and returns a tuple `(minimum, maximum, mean, median)`. The minimum and maximum values can be integers, but mean and median must be returned as floats.
#  
# Tips:
# 
# - If there is an even number of values in the list, there is no single middle value; in this case, take the median to be the mean of the two middle values.
# - Use [sorted()](https://docs.python.org/3/library/functions.html#sorted) to handle an unordered list. See below.
# - The mean and the median should be floating values. To convert an integer to a float, use [float()](https://docs.python.org/3/library/functions.html#float).

# In[3]:

def get_stats(input_list):
    '''
    Accepts a list of integers, and returns a tuple of four numbers:
    minimum(int), maximum(int), mean(float), and median(float)
    
    Parameters
    __________
    input_list: A list of integers.
      
    Returns
    _______
    minimum: An integer. The minimum of input_list.
    maximum: An integer. The maximum of input_list.
    mean: A float. The mean of input_list.
    median: A float. The median of input_list.
            If there is an even number of values in the list,
            there is no single middle value.
            In this case, take the median to be the mean of the two middle values.
    
    Examples
    ________
    >>> get_stats([0, 1, 2, 3, 4])
    (0, 4, 2.0, 2.0)
    >>> get_stats([0, 1, 2, 3, 4, 5])
    (0, 5, 2.5, 2.5)
    >>> get_stats([0, 1, 2, 5])
    (0, 5, 2.0, 1.5)
    >>> get_stats([2, 0, 4, 1, 5])
    (0, 5, 2.4, 2.0)
    '''
    
    # your code goes here
    list_sort = sorted(input_list)
    minimum = min(list_sort)
    maximum = max(list_sort)
    N = len(list_sort)
    mean = float(sum(list_sort)/N)
    if(N%2 == 1):
        median = float(list_sort[int(N/2)])
    else:
        median = float(0.5 * (list_sort[int(N/2)-1] + list_sort[int(N/2)]))
      

    return minimum, maximum, mean, median


# In[8]:

get_stats(list(range(51)))


# When you are done writing the `get_stats()` function, test your function by
# running the following cells.
# 
# We first test a list with an even number integers. For `list(range(51))`, your output should be
# 
#     Minimum = 0 is of type <class 'int'>.
#     Maximum = 50 is of type <class 'int'>.
#     Mean = 25.0 is of type <class 'float'>.
#     Median = 25.0 is of type <class 'float'>.

# In[11]:

my_list = list(range(51)) # from 0 to 50. an odd number of elements.
min_val, max_val, mean_val, med_val = get_stats(my_list)
print("Minimum = {0} is of type {1}.".format(min_val, type(min_val)))
print("Maximum = {0} is of type {1}.".format(max_val, type(max_val)))
print("Mean = {0:.1f} is of type {1}.".format(mean_val, type(mean_val)))
print("Median = {0:.1f} is of type {1}.".format(med_val, type(med_val)))


# We also test a list with an odd number of elements. For `list(range(52))`, the output should be
# 
#     Minimum = 0 is of type <class 'int'>.
#     Maximum = 51 is of type <class 'int'>.
#     Mean = 25.5 is of type <class 'float'>.
#     Median = 25.5 is of type <class 'float'>.

# In[12]:

another_list = list(range(52)) # from 0 to 51, an even number of elements.
min_val, max_val, mean_val, med_val = get_stats(another_list)
print("Minimum = {0} is of type {1}.".format(min_val, type(min_val)))
print("Maximum = {0} is of type {1}.".format(max_val, type(max_val)))
print("Mean = {0:.1f} is of type {1}.".format(mean_val, type(mean_val)))
print("Median = {0:.1f} is of type {1}.".format(med_val, type(med_val)))


# Your function should also be able to handle a *shuffled* list. (Hint: Use [sorted()](https://docs.python.org/3/library/functions.html#sorted) in `get_stats()`.)

# In[13]:

from random import shuffle
shuffle(my_list)
print(my_list)


# Running the following cell should give the same output as the case when `my_list = list(range(51))`:
# 
#     Minimum = 0 is of type <class 'int'>.
#     Maximum = 50 is of type <class 'int'>.
#     Mean = 25.0 is of type <class 'float'>.
#     Median = 25.0 is of type <class 'float'>.

# In[14]:

min_val, max_val, mean_val, med_val = get_stats(my_list)
print("Minimum = {0} is of type {1}.".format(min_val, type(min_val)))
print("Maximum = {0} is of type {1}.".format(max_val, type(max_val)))
print("Mean = {0:.1f} is of type {1}.".format(mean_val, type(mean_val)))
print("Median = {0:.1f} is of type {1}.".format(med_val, type(med_val)))


# In[ ]:



