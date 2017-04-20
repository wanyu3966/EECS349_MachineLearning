# EECS 349 Machine Learning
# HW2
# @Author: Liangji Wang

import csv #this would be for reading our text files
import numpy as np #this would be for computing levenshtein_distance (multi-dimensional arrays are easier with these)
import matplotlib.pyplot as plt #this would be for making many of the plots required in this assignment
import sys
import time
import random
from mpl_toolkits.mplot3d import axes3d
import math
from matplotlib import cm

# data:
# Each line in wikipediatypo.txt contains a common misspelled word followed by its associated correction. The two words (or phrases) on each line (error, correction) are separated by a tab. Note, some corrections may contain blanks (e.g. a line containing aboutto and about to).
# The file wikipediatypoclean.txt is a subset of wikipediatypo.txt that contains only words whose correct spelling is an entry in the dictionary file /American/3esl.txt from the 12dicts data set. It is in the same format as wikipediatypo.txt. 
# The file syntheticdata.txt has the same format as the other two files, but the spelling errors were created using a synthetic distribution. 

# read single col file, in this hw is the dictionary file
def readSingleCol(path):
    dataList = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataList.append(row[0].lower())
    return dataList

# read double col file, in this hw are the wikitypo, wikitypoclean files
def readDoubleCol(path):
    fstList = []
    sndList = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile,delimiter='\t')
        for row in reader:
            fstList.append(row[0].lower())
            sndList.append(row[1].lower())
    return fstList, sndList

# open saved result file and turn it to the error rate dictionary
# for testing and visualize result
def readResult(path):
    dataDict = {}
    with open(path) as csvfile:
        reader = csv.reader(csvfile,delimiter='\t')
        for row in reader:
            #print row
            key = str(row[0] + '\t' + row[1] + '\t' + row[2])
            dataDict[key] = float(row[3])
    return dataDict

# write file with two col
def writeToFile(col1, col2, fileName = None):
    if fileName is None:
        fileName = 'corrected.txt'
    with open(fileName, 'w') as myfile:
        #writer = csv.writer(csvfile)
        length = len(col1)
        for i in range(length):
            line = str(col1[i]) + '\t' + str(col2[i]) + '\n'
            myfile.write(line)
    return


#write some code to do this, calling levenshtein_distance, and return a string (the closest word)
def find_closest_word (string1, dictionary, deletion_cost = None, insertion_cost = None, substitution_cost = None, qweFlag = None):
    if deletion_cost is None:
        deletion_cost = 1
    if insertion_cost is None:
        insertion_cost = 1
    if substitution_cost is None:
        substitution_cost = 1
    if qweFlag is None:
        qweFlag = 0
    length = sys.maxint
    correctedList = []
    correctedList.append(string1)
    for word in dictionary:
        if qweFlag == 0:
            temp = levenshtein_distance(string1, word, deletion_cost, insertion_cost, substitution_cost)
        else:
            temp = qwerty_levenshtein_distance(string1, word, deletion_cost, insertion_cost)
        if temp == length:
            correctedList.append(word)
        if temp < length:
            correctedList[:] = []
            correctedList.append(word)
            length = temp
    corrected = correctedList[random.randint(0, len(correctedList) - 1)]

    return corrected

#write some code to compute the levenshtein distance between two strings, given some costs, return the distance as an integer
def levenshtein_distance(string1, string2, deletion_cost, insertion_cost, substitution_cost):
    m = len(string1) + 1
    n = len(string2) + 1
    d = np.zeros((m, n))
    for i in range(0, m): 
        d[i, 0] = i * deletion_cost
    for j in range(0, n):
        d[0, j] = j * insertion_cost
    for j in range(1, n):
        for i in range(1, m):
            if string1[i-1] == string2[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j] + deletion_cost, min(d[i, j-1] + insertion_cost, d[i-1, j-1] + substitution_cost))
    return d[m-1, n-1]

def qwertyCharLength(char1, char2):
    qweKeyboard = {'q': {'row':0, 'col':0}, 'w': {'row':1, 'col':0}, 'e': {'row':2, 'col':0}, 'r': {'row':3, 'col':0}, 't': {'row':4, 'col':0}, 'y': {'row':5, 'col':0}, 'u': {'row':6, 'col':0}, 'i': {'row':7, 'col':0}, 'o': {'row':8, 'col':0}, 'p': {'row':9, 'col':0}, 'a': {'row':0, 'col':1},'z': {'row':0, 'col':2},'s': {'row':1, 'col':1},'x': {'row':1, 'col':2},'d': {'row':2, 'col':1},'c': {'row':2, 'col':2}, 'f': {'row':3, 'col':1}, 'b': {'row':4, 'col':2}, 'm': {'row':6, 'col':2}, 'g': {'row':4, 'col':1}, 'h': {'row':5, 'col':1}, 'j': {'row':6, 'col':1}, 'k': {'row':7, 'col':1}, 'l': {'row':8, 'col':1}, 'v': {'row':3, 'col':2}, 'n': {'row':5, 'col':2} }
    if char1.isalpha() and char2.isalpha():
        return abs(qweKeyboard[char1]['row'] - qweKeyboard[char2]['row']) + abs(qweKeyboard[char1]['col'] - qweKeyboard[char2]['col'])
    # if input is not alphabet, output 9
    # need to be fixed?
    return 9

def qwerty_levenshtein_distance(string1, string2, deletion_cost = None, insertion_cost = None):
    if deletion_cost is None:
        deletion_cost = 1
    if insertion_cost is None:
        insertion_cost = 1
    m = len(string1) + 1
    n = len(string2) + 1
    d = np.zeros((m, n))
    #print string1, string2
    for i in range(0, m): 
        d[i, 0] = i * deletion_cost
    for j in range(0, n):
        d[0, j] = j * insertion_cost
    for j in range(1, n):
        for i in range(1, m):
            if string1[i-1] == string2[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j] + deletion_cost, min(d[i, j-1] + insertion_cost, d[i-1, j-1] + qwertyCharLength(string1[i-1], string2[j-1])))
    return d[m-1, n-1]

#find whether the corrected typo using the dictionary matches the true word. 0 if it doesn't, 1 if it does. Count them all up and return a real value between 0 and 1 representing the error_rate
def measure_error(typos, truewords, dictionarywords):
    start = time.time()
    correctedList = []
    for typo in typos:
        correctedList.append(find_closest_word(typo, dictionarywords))
    length = len(correctedList)
    trueCount = 0
    for i in range(length):
        trueList = truewords[i].split(', ')
        for trueWord in trueList:
            if correctedList[i] == trueWord:
                trueCount += 1
                #print trueWord
                break
    error_rate = float(length - trueCount) / length
    print 'time past: ' + str(time.time() - start) + '\terror_rate: ' + str(error_rate) 
    return error_rate

# for problem 3 & 4
# this function will also do the mearsure rate for valuePool of [0, 1, 2, 4]
# flag 0 for normal levenshtein_distance
# flag not 0 for qwerty levenshtein_distance
def problem34(typos, truewords, dictionarywords, flag = None):
    if flag is None:
        flag = 0
    valuePool = [0, 1, 2, 4]
    rateDict = {}
    for a in valuePool:
        for b in valuePool:
            for c in valuePool:
                expName = str(a) +'\t'+ str(b) +'\t'+ str(c)
                start = time.time()
                correctedList = []
                for typo in typos:
                    correctedList.append(find_closest_word(typo, dictionarywords, a, b, c, flag))
                length = len(correctedList)
                trueCount = 0
                for i in range(length):
                    trueList = truewords[i].split(', ')
                    for trueWord in trueList:
                        if correctedList[i] == trueWord:
                            trueCount += 1
                            #print trueWord
                            break
                #writeToFile(correctedList, truewords, 'exp_' + expName + '.txt')
                error_rate = float(length - trueCount) / length
                rateDict[expName] = error_rate
                print 'exp', expName, 'time past:', time.time() - start, '\terror rate:', error_rate
                if flag != 0:
                    break
    return rateDict

# this function will do subset of typos, truewords, and dictionary

# flag 0 for normal levenshtein_distance
# flag not 0 for qwerty levenshtein_distance
def doExp(typos, truewords, dictionarywords, flag, testSize, dictSize):
    timeStart = time.time()
    # wordList[random.randint(0, len(wordList) - 1)] 
    subTypos = []
    subTrue = []
    subDict = []
    if testSize >= len(typos):
        subTypos = typos
        subTrue = truewords
    else:
        for i in range(testSize):
            index = random.randint(0, len(typos) - 1)
            subTypos.append(typos.pop(index))
            subTrue.append(truewords.pop(index))
    #print 'sub typos are:\n', subTypos
    #print 'sub truewords are:\n', subTrue
    if dictSize >= len(dictionarywords):
        subDict = dictionarywords
    else:
        for i in range(dictSize):
            subDict.append(dictionarywords.pop(random.randint(0, len(dictionarywords) - 1)))
        for item in subTrue:
            if item not in subDict:
                subDict.append(item)   
    for item in subDict:
        if not item.isalpha():
            subDict.remove(item)
    #writeToFile(subTypos, subTrue, 'subTypoTrue.txt')
    rateDict = problem34(subTypos, subTrue, subDict, flag)
    print 'experiment time cost:', str(time.time() - timeStart)
    return rateDict

# get all list information, return the list combination used for plot's X, Y, Z
def plotHelper(deletionList, insertionList, substitutionList, rateList, subValue):
    X = []
    Y = []
    Z = []
    #print deletionList, insertionList, substitutionList
    for i in range(len(substitutionList)):
        if int(substitutionList[i]) == int(subValue):
            X.insert(-1, int(deletionList[i]))
            Y.insert(-1, int(insertionList[i]))
            Z.insert(-1, rateList[i])
    return X, Y, Z

# read error rate dictionary and plot it with trisurf or scatter
# combined with readResult function that can open saved result file and plot it
def resultPlot(dataDict, flag = None):
    deletionList = []
    insertionList = []
    substitutionList = []
    rateList = []
    if flag is None:
        flag = 0
    for key in dataDict:
        valueList = key.split()
        #print valueList
        deletionList.insert(-1, valueList[0])
        insertionList.insert(-1, valueList[1])
        rateList.insert(-1, dataDict[key])
        if flag == 0:
            substitutionList.insert(-1, valueList[2])
        else:
            substitutionList.insert(-1, 0)
    fig = plt.figure()
    if flag == 0:
        subFigList = []
        X = {}
        Y = {}
        Z = {}
        for i in range(4):
            subV = int(math.pow(2, i)) / 2
            subVstring = str(subV)
            #print subV
            subFigList.insert(0, fig.add_subplot(221 + i, projection = '3d'))
            #subFigList.insert(0, fig.gca(projection = '3d'))
            subFigList[0].set_title('problem 3, substitution =' + str(subVstring))
            subFigList[0].set_xlabel('deletion_cost')
            subFigList[0].set_ylabel('insertion_cost')
            subFigList[0].set_zlabel('error_rate')
            X[subVstring], Y[subVstring], Z[subVstring] = plotHelper(deletionList, insertionList, substitutionList, rateList, subV)
            #print 'x', X[subVstring]
            #print 'y', Y[subVstring]
            #print 'z', Z[subVstring]
            subFigList[0].set_xlim(0, 4)
            subFigList[0].set_ylim(0, 4)
            subFigList[0].set_zlim(0, 1)
            subFigList[0].view_init(elev=12, azim=40)              # elevation and angle
            subFigList[0].dist=12 
        #print X, Y, Z
            #subFigList[0].scatter(X[subVstring], Y[subVstring], Z[subVstring], 
             #   color='purple',                            # marker colour
              #  marker='o',                                # marker shape
               # s=30                                       # marker size
            #)
            subFigList[0].plot_trisurf(X[subVstring], Y[subVstring], Z[subVstring], cmap=cm.jet, linewidth=0.1)
    else:
        sub0 = fig.add_subplot(111, projection = '3d')
        sub0.set_title('problem 4\nsubstitution = qwertyKeyBoardManhattanDistance')
        sub0.set_xlabel('deletion_cost')
        sub0.set_ylabel('insertion_cost')
        sub0.set_zlabel('error_rate')
        x, y, z = plotHelper(deletionList, insertionList, substitutionList, rateList, 0)
        sub0.set_xlim(0, 4)
        sub0.set_ylim(0, 4)
        sub0.set_zlim(0, 1)
        sub0.view_init(elev=12, azim=40)              # elevation and angle
        sub0.dist=12 
        #print x, y, z
        #sub0.scatter(x, y, z, 
         #  color='purple',                            # marker colour
          # marker='o',                                # marker shape
           #s=30                                       # marker size
           #)
        sub0.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    plt.show()

# read whole line, for problem 2b
def readLine(path):
    typoList = []
    with open(path) as f:
        for line in f:
            typoList.append(line)
    return typoList

# write whole line, for problem 2b
def writeLine(lineList):
    with open('corrected.txt', 'w') as myfile:
        for i in range(len(lineList)):
            myfile.write(lineList[i])
    pass

# do the problem 2b
def problem2b(pathToBeChecked, pathDict):
    typoList = readLine(pathToBeChecked)
    dictList = readSingleCol(pathDict)
    correctedList = []
    for i in range(len(typoList)):
        word = ''
        typoLine = typoList[i]
        correctedLine = ''
        for j in range(len(typoLine)):
            if typoLine[j].isalpha():
                word += typoLine[j]
            else:
                if word != '':
                    word = find_closest_word(word, dictList, 1, 1, 1, 0)
                correctedLine = correctedLine + word + typoLine[j]
                word = ''
        correctedLine += word
        #print correctedLine
        correctedList.append(correctedLine)
    writeLine(correctedList)
    pass

def problem2c(pathToBeChecked, pathDict):
    typoList, correctList = readDoubleCol(pathToBeChecked)
    dictList = readSingleCol(pathDict)
    return measure_error(typoList, correctList, dictList)

def twoExp(pathToBeChecked, pathDict):
    # flag 0 for normal levenshtein_distance
    # flag not 0 for qwerty levenshtein_distance
    typoList, correctList = readDoubleCol(pathToBeChecked)
    dictList = readSingleCol(pathDict)
    for i in range(2):
        flag = i
        rateDict = doExp(typoList, correctList, dictList, flag, 80, 4000)
        # can call write result here
        resultPlot(rateDict, flag)
    #example of write result and read result
    #write:
    #flag = 0
    #keyList = []
    #    rateList = []
    #    for key in rateDict:
    #        keyList.append(key)
    #        rateList.append(rateDict[key])
    #writeToFile(keyList, rateList, 'expResult'+ str(flag) +'.txt')
    #read:
    #flag = 0
    #rateDict = readResult('expResult'+ str(flag) +'.txt')




def main():
    argv = sys.argv
    print 'Number of arguments:', len(argv), 'arguments.'
    print 'Argument List:', argv
    pathToBeChecked = str(argv[1])
    pathDict = str(argv[2])
    problem2b(pathToBeChecked, pathDict)

    #pathToBeChecked = "wikipediatypoclean.txt"
    #pathDict = "3esl.txt"
    #problem2c(pathToBeChecked, pathDict)
    #twoExp(pathToBeChecked, pathDict)

    return 0

if __name__ == "__main__":
    sys.exit(main())

# Unused function (might be useful later)

#def readMultiCol(path):
 #   dataList = []
  #  with open(path) as csvfile:
   #     reader = csv.reader(csvfile,delimiter='\t')
    #    for row in reader:
     #       dataList.append(row)
    #return dataList

#def readAsDict(path):
 #   dataList = []
  #  with open(path) as csvfile:
   #     reader = csv.DictReader(csvfile, ("wrong", "right"), delimiter='\t')
    #    for row in reader:
            #print row['right'], type(row['right'])
     #       dataList.append(row)
    #return dataList

#def stringToWordList(string1):
#    parts = [''.join(c for c in s if c.isalpha()) for s in string1.split()]
#    return parts

#def insert (source_str, insert_str, pos):
#    return source_str[:pos]+insert_str+source_str[pos:]

