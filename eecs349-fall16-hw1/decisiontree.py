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

# read one file from file path,
# return the data list of dict with attribution as key value
def readFile(path):
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        dataList = []
        for row in reader:
            #print(row)
            dataList.append(row)
        # this double loop below will change value of string 'true' / 'false'
        # to bool type
        for item in dataList:
            for key in item:
                if item[key] == 'true':
                    item[key] = True
                else:
                    item[key] = False
        #print dataList[0]
    return dataList

# pseudo-random pick the training data set and test data set
# after this function, data list is split, which means changed
def getTrainingSet(dataList, trainingSetSize):
    dataSize = len(dataList)
    if trainingSetSize <= 0:
        print 'trainingSetSize cannot <= 0'
        sys.exit()
    if trainingSetSize >= dataSize:
        print 'trainingSetSize cannot >= listLength'
        sys.exit()
    trainingList = []
    for i in range(0, trainingSetSize):
        trainingList.append(dataList.pop(random.randint(0, dataSize - 1)))
        dataSize -= 1
    return trainingList, dataList

# get key value from the instance dict in datalist
# return a list of key value without classification
def getKeyList(dataList):
    keyList = []
    for key in dataList[0]:
        if key != 'CLASS':
            keyList.append(key)
    return keyList

# pass datalist and attribution, return the entropy value of the specific attribute
# since the values of each attribute for each instance are only true/false,
# suppose P(X == true) = p, thus H(X) = -pLog(p) - (1-p)Log(1-p)
def calEntropy(dataList, key):
    #print 'in entropy function'
    trueCount = 0
    for instance in dataList:
        if instance[key] == True:
            trueCount += 1
    if trueCount == 0 or trueCount == len(dataList):
        return 0
    p = float(trueCount) / float(len(dataList))
    return -p * float(math.log(p, 2)) - float(1-p) * float(math.log(1-p, 2))

# calculate Gain value, will call calEntropy
def calGain(dataList, key):
    #print 'gain function'
    eS = calEntropy(dataList, 'CLASS')
    trueList = []
    falseList = []
    for instance in dataList:
        if instance[key] == True:
            trueList.append(instance)
        else:
            falseList.append(instance)
    eT = float(len(trueList))/len(dataList) * calEntropy(trueList, 'CLASS')
    eF = float(len(falseList))/len(dataList) * calEntropy(falseList, 'CLASS')
    #print 'es, et, ef', eS, eT, eF
    return eS - eT - eF

# from the exist data list, choose the best attribute
# return the attList after pick and best attribute
def attributeChoose(dataList, attList):
    bestAtt = ''
    gainValue = -1
    for key in attList:
        gain = calGain(dataList, key)
        if gainValue < gain:
            gainValue = gain
            bestAtt = key
    return bestAtt

# check all examples whether have the same classification,
# return true for all same classification, otherwise return false
def checkClass(dataList):
    classValue = dataList[0].get('CLASS')
    for instance in dataList:
        if classValue != instance.get('CLASS'):
            return False
    return True

# return most significant value
def Mode(dataList):
    trueValue = 0
    for instance in dataList:
        if instance.get('CLASS') == True:
            trueValue += 1
    return trueValue > (len(dataList) / 2)

# class Node
class Node:
    # in the subDict, key is the sub node label, item is the subNode
    def __init__(self, attribute):
        self.attribute = attribute
        self.subDict = {}
        # parent is root by default
        self.parent = 'root'
    def addParent(self, parent):
        self.parent = parent
    def addSub(self, label, subNode):
        self.subDict[label] = subNode

# split data list to subset depend on key and value
def dataSplit(dataList, key, value):
    splitList = []
    for instance in dataList:
        if instance[key] == value:
            index = dataList.index(instance)
            splitList.append(dataList[index])
    return splitList

# get values from best attribute
def getValueList(dataList, best):
    valueList = []
    for instance in dataList:
        if instance.get(best) not in valueList:
            valueList.append(instance.get(best))
    return valueList

# ID3 algorithm to create tree
def DTL(dataList, attributes, default):
    if not dataList:
        return default
    elif checkClass(dataList):
        return dataList[0].get('CLASS')
    elif not attributes:
        return Mode(dataList)
    else:
        best = attributeChoose(dataList, attributes)
        tree = Node(best)
        for value in getValueList(dataList, best):
            examples = dataSplit(dataList, best, value)
            if best in attributes:
                attributes.remove(best)
            subTree = DTL(examples, attributes, Mode(examples))
            if type(subTree) is not bool:
                subTree.addParent(best)
            tree.addSub(value, subTree)
        return tree

# print tree structure
def treePrint(root):
    if type(root) is not bool:
        subleaf = ''
        for key in root.subDict:
            if type(root.subDict[key]) is not bool:
                subleaf = subleaf + str(key) + ':' + root.subDict[key].attribute + ' '
            else:
                subleaf = subleaf + str(key) + ':' + str(root.subDict[key]) + ' '
        print 'parent:', root.parent, '; attribute:', root.attribute, '; subLeaf:', subleaf
        for key in root.subDict:
            if type(root.subDict[key]) is not bool:
                treePrint(root.subDict[key])
    else:
        print 'special case for root is a bool:', root

# helper function to test decision tree
def testTreeHelper(example, node):
    if type(node) is bool:
        return node
    else:
        currentAttribute = node.attribute
        key = example[currentAttribute]
        if key not in node.subDict:
            return -1
        subNode = node.subDict[key]
        if subNode == None:
            return -1
            pass
        elif type(subNode) is bool:
            return subNode
        else:
            return testTreeHelper(example, subNode)

# tstTree function, test the tree generated by DTL with testing list
def testTree(testingList, tree, verbose):
    correctCount = 0
    for instance in testingList:
        value = testTreeHelper(instance, tree)
        #if value != -1:
        if type(value) is bool:
            #print value
            if instance['CLASS'] == value:
                correctCount += 1
            else:
                if verbose == 1:
                    print 'this example %s is failed in tree' %(instance)
        else:
            #print '--------------1111111111111111', value
            if verbose == 1:
                print 'this example %s is failed in tree' %(instance)
    return correctCount / float(len(testingList))

# get prior probability only for True case
def getPriorPro(trainingList, testingList, verbose):
    count = 0
    value = Mode(trainingList)
    for instance in testingList:
        if instance.get('CLASS') == value:
        #if instance.get('CLASS') == True:
            count += 1
        else:
            if verbose == 1:
                print 'this example %s is failed in Prior probability' %(instance)
    return count / float(len(testingList))

# print item in list line by line
def listPrint(dataList):
    for item in dataList:
        print item

# main function goes here
def main():
    argv = sys.argv
    print 'Number of arguments:', len(argv), 'arguments.'
    print 'Argument List:', argv
    path = str(argv[1])
    trainingSetSize = int(argv[2])
    numberOfTrials = int(argv[3])
    verbose = int(argv[4])
    meanByTree = 0
    meanByPrior = 0
    print '-------------------------------------------------'
    for i in range(0, numberOfTrials):
        print 'trial number:', i + 1
        print '------------------------------------------------'
        dataList = readFile(path)
        keyList = getKeyList(dataList)
        trainingList, testingList = getTrainingSet(dataList, trainingSetSize)
        if verbose == 1:
            print 'training list is :\n'
            listPrint(trainingList)
            print '\ntesting list is:\n'
            listPrint(testingList)
        tree = DTL(trainingList, keyList, Mode(trainingList))
        treePrint(tree)
        testTreeValue = testTree(testingList, tree, verbose) * 100
        meanByTree += testTreeValue 
        priorValue = getPriorPro(trainingList, testingList, verbose) * 100
        meanByPrior += priorValue 
        print 'percent of test cases correctly classified by using prior probabilities of true from the traning set = %.2f%% \npercent of test cases correctly classified by a decision tree built with ID3 = %.2f%%' %(priorValue, testTreeValue)
        print '-------------------------------------------------\ntrial:', i + 1, 'ends\n-------------------------------------------------'
    print 'number of training list is: %d\nnumber of testing list is: %d\nnumber of trial is: %d\nexample file path: %s read as csv file\nmean performance of using prior probability of true derived from the training set = %.2f%%\nmean performance of decision tree over all trials = %.2f%%' %(len(trainingList), len(testingList), numberOfTrials, path, meanByPrior/numberOfTrials, meanByTree/numberOfTrials)
    return 0

if __name__ == "__main__":
    sys.exit(main())

