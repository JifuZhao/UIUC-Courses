#!/usr/bin/env python3

import sys

# YOUR CODE HERE
sep = '\t'

with sys.stdin as fin:
    with sys.stdout as fout:
        # keep tracking the current status
        value = []
        word = None
        current_word = None
        
        # loop through every line in Stdin
        for line in fin:
            line = line.strip()
            word, DepDelay = line.split('\t', 1)
            
            if current_word == word:
                if DepDelay != 'NA':
                    value.append(int(DepDelay))
            else:
                if current_word != None:
                    fout.write('{0}{1}{2}{1}{3}\n'.format(current_word, sep, min(value), max(value)))
                
                # new word
                current_word = word
                value = []
                if DepDelay != 'NA':
                    value.append(int(DepDelay))
        else:
            # output the final result
            if current_word == word:
                fout.write('{0}{1}{2}{1}{3}\n'.format(current_word, sep, min(value), max(value)))