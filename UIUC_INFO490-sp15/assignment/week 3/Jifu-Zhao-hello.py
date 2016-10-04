# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Problem 3.1. The obligatory "hello world" problem.

# <markdowncell>

# Write a very simple program that asks the user to enter his or her name,
#  and prints out a welcome message that is customized to the user's name.
#  For example, the program should wait for the user's input after printing out
# ```console
# Enter your name: 
# ```
# When you enter your name,
# ```console
# Enter your name: world
# ```
# the output on the next line should be
# ```console
# Hello, world!
# ```

# <markdowncell>

# First, define a function named `welcome()`.

# <codecell>

def welcome():
    '''
    Ask user for name and print a welcome message customized to the person's name.
    
    Examples:
    >>> welcome()
    Enter your name: World
    Hello, World!
    '''
    
    # your code goes here.
    name = str(input("Enter your name: "))
    name = name + "!"
    print ("Hello,", name)
    
    return None

# <markdowncell>

# If you have finished writing the function `welcome()`, run the cell below to check your answer.

# <codecell>

welcome()

# <codecell>


