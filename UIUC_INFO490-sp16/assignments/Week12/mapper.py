#!/usr/bin/env python3

import sys

# YOUR CODE HERE
sep = '\t'
i = 0

with sys.stdin as fin:
    with sys.stdout as fout:
        # read the data from fin
        for line in fin:
            # skip the first line
            if i == 0:
                i = 1
                continue
            # find the useful information    
            line = line.strip()
            words = line.split(',')
            Origin = words[16]
            DepDelay = words[15]
            
            # output the result
            fout.write('{0}{1}{2}\n'.format(Origin, sep, DepDelay))
        