#!/usr/bin/python

# @Author: Liangji Wang

# Usage: 
# python decisiontree.py <inputFileName> <trainingSetSize> numberOfTrials> <verbose>
# inputFileName - trainingSetSize - numberOfTrials - verbose -
# the fully specified path to the input file. Note that windows pathnames may require double backslash \\
# an integer specifying the number of examples from the input file that will be used to train the system
# an integer specifying how many times a decision tree will be built from a randomly selected subset of the training examples.
# a string that must be either 1 or 0
# If verbose is 1 the output will include the training and test sets. Else the output will only contain a description of the tree structure and the results for the trials.

import sys
import math
import csv
import random


def calEntropy(pos, nega):
    if pos == 0 or nega == 0:
        return 0
    p = float(pos) / float(pos + nega)
    return -p * float(math.log(p, 2)) - float(1-p) * float(math.log(1-p, 2))

print calEntropy(1, 2)
