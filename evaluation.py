#!/usr/bin/python
from __future__ import division
import re
import nltk
import sys
import getopt
import pickle
import time
import statistics
import matplotlib.pyplot as plt

def usage():
    print("usage: " + sys.argv[0] + "-o output-file-of-results -r file-of-relevant")

file_of_output = file_of_relevant = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'o:r:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-o':
        file_of_output = a
    elif o == '-r':
        file_of_relevant = a
    else:
        assert False, "unhandled option"

if file_of_output == None or file_of_relevant == None:
    usage()
    sys.exit(2)

def evaluate(correctList, actualResult):
    print(correctList)
    correctSet = set(correctList)
    relevant = 0
    F2List = []
    precisionList = []
    recallList = []
    MAPList = []
    for x in range(len(actualResult)):
        result = int(actualResult[x])
        if result in correctSet:
            relevant += 1
            precision = relevant/(x + 1)
            recall = relevant/len(correctList)
            precisionList.append(precision)
            recallList.append(recall)
            F2List.append((5*precision*recall)/(5*precision + recall))
            AF2 = statistics.mean(F2List)
            print("Position: " + str(x))
            print("AF2: " + str(AF2))
            precision = statistics.mean(precisionList)
            print("MAP: " + str(precision))
            print("R-precision: " + str(precision))
            print("===================")
    plt.plot(recallList, precisionList)
    # naming the x axis
    plt.xlabel('Recall')
    # naming the y axis
    plt.ylabel('Precision')

    # giving a title to my graph
    plt.title('Precision-Recall Curve')

    #mngr = plt.get_current_fig_manager()
    #mngr.window.setGeometry(0,0,1, 1)
    # function to show the plot
    plt.show()

relevantFile = open(file_of_relevant, "r")
query = relevantFile.readline()
correctList = []
for correctResult in relevantFile:
    correctList.append(int(correctResult))
#search.runSearch(dict_file, postings_file, queries_file, results_file)

output = open(file_of_output, "r")
result = output.readline()
resultsList = result.split(" ")

evaluate(correctList, resultsList)
