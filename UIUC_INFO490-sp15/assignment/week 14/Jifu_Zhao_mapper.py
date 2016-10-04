#!/usr/bin/env python3

import sys

# We explicitly define the word/count separator token.
sep = '\t'
i = 0

# We open STDIN and STDOUT
with sys.stdin as fin:
    with sys.stdout as fout:
    
        # For every line in STDIN
        for line in fin:
            if i == 0:
                i = 1
                continue
        
            # Strip off leading and trailing whitespace
            line = line.strip()
            
            # We split the line into word tokens. Use whitespace to split.
            # Note we don't deal with punctuation.
            
            words = line.split(',')[16]
            
            # Now loop through all words in the line and output

            fout.write("{0}{1}1\n".format(words, sep))