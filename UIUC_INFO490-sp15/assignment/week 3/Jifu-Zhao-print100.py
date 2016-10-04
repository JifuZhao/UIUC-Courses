# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## Problem 3.2. A Famous Simple Interview Question.
# 
# This problem is a famous programming job interview question designed to filter out those who think they can program but in reality can't. If you can easily solve this problem, you are already better than 99% of job candidates who apply for a programming job. The problem statement is simple:
# 
#  - Print every number from 1 to 100.
#  - If the number is a multiple of 4, print "info" instead.
#  - If the number is a multiple of 6, print "matics" instead.
#  - If the number is a multiple of 4 **and** 6, print "informatics" instead.
#  
# Note the spelling of "info**r**matics". There's an R in informatics.

# <markdowncell>

# Use conditionals and `str()` function to write a function named `get_informatics()` that takes an integer and returns a string. I emphasize that the return value is of type `str`, not `int`.

# <codecell>

def get_informatics(number):
    '''
    Prints every number from 1 to 100. But for multiples of four, print "info" instead;
    and for multiples of six, print "matics." For numbers which are mltiples of both
    four and six, print "informatics."
    
    Parameters
    __________
    input: An integer greater than 0.
    
    Returns
    _______
    output: A string.
    
    Examples
    ________
    >>> get_informatics(1)
    '1'
    >>> get_informatics(4)
    'info'
    >>> get_informatics(6)
    'matics'
    >>> get_informatics(12)
    'informatics'
    '''
    
    # your code goes here
    a = number % 4
    b = number % 6
    if(a == 0 and b ==0):
        result = "informatics"
    elif(a == 0 and b != 0):
        result = "info"
    elif(a != 0 and b == 0):
        result = "matics"
    else:
        result = str(number)
        
    return result

# <markdowncell>

# Next, use `for` loop(s) and `get_informatics()` function to write a function named `print100()` that prints every number from 1 to 100.

# <codecell>

def print100():
    '''
    Prints every number from 1 to 100, but multiples of 4, 6, and both 4 and 6
    are replaced by 'info', 'matics', and 'informatics', respectively.
    '''
    
    # your code goes here
    for number in range(1,101):
        c = get_informatics(number)
        print(c)
        
    return None

# <markdowncell>

# When you are done writing your functions, test the cell below by pressing <kbd>shift</kbd> + <kbd>enter</kbd>.

# <codecell>

print100()

# <markdowncell>

# When you execute `print_informatics()`, your output should be
# 
# ```text
# 1
# 2
# 3
# info
# 5
# matics
# 7
# info
# 9
# 10
# 11
# informatics
# 13
# 14
# 15
# info
# 17
# matics
# 19
# info
# 21
# 22
# 23
# informatics
# 25
# 26
# 27
# info
# 29
# matics
# 31
# info
# 33
# 34
# 35
# informatics
# 37
# 38
# 39
# info
# 41
# matics
# 43
# info
# 45
# 46
# 47
# informatics
# 49
# 50
# 51
# info
# 53
# matics
# 55
# info
# 57
# 58
# 59
# informatics
# 61
# 62
# 63
# info
# 65
# matics
# 67
# info
# 69
# 70
# 71
# informatics
# 73
# 74
# 75
# info
# 77
# matics
# 79
# info
# 81
# 82
# 83
# informatics
# 85
# 86
# 87
# info
# 89
# matics
# 91
# info
# 93
# 94
# 95
# informatics
# 97
# 98
# 99
# info
# ```

# <codecell>


