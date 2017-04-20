# EECS 349 Machine Learning
# HW4
# @Author: Liangji Wang

import csv #this would be for reading our text files
import sys
import numpy as np
import copy
import json

import scipy
import matplotlib.pyplot as plt

# read double col file, in this hw are the wikitypo, wikitypoclean files
def readUData(path):
    list1 = []
    list2 = []
    #list3 = []
    #list4 = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile,delimiter='\t')
        for row in reader:
            list1.insert(0, int(row[0]))
            list2.insert(0, int(row[1]))
            #list3.insert(0, int(row[2]))
            #list4.insert(0, int(row[3]))
    return list1, list2#, list3, list4

def getUserDict(uidList, itemIdList):
    # key is uid, value is list of watched movie id 
    userDict = {}
    for i in range(len(uidList)):
        uid = uidList[i]
        movieId = itemIdList[i]
        if uid in userDict:
            userDict[uid].insert(0, movieId)
        else:
            userDict[uid] = [movieId]
    return userDict

def addToMovieDict(movieDict, movieId):
    if movieId in movieDict:
        counter = movieDict[movieId]
        movieDict[movieId] = counter + 1
    else:
        print 'undefined key value:', movieId
        movieDict[movieId] = 1

#def findCommonMovie(commonMovieDict, user1, user2):
#    for u1_movie in user1[0]:
#        for u2_movie in user2[0]:
#            if u1_movie == u2_movie:
#                addToMovieDict(commonMovieDict, u1_movie)
#                break

def findMovieCount(movieCountDict, user):
    for movieId in user:
        addToMovieDict(movieCountDict, movieId)

def findCommonView(commonViewDict, user1, user2, userId1, userId2):
    counter = 0
    for u1_movie in user1:
        for u2_movie in user2:
            if u1_movie == u2_movie:
                counter += 1
                break
    key = str(userId1) + 'and' + str(userId2)
    if key in commonViewDict:
        print 'exist key'
    else:
        commonViewDict[key] = counter

def getCommonDict(userDict):
    # key is movie id, value is view counter
    compareDict = copy.deepcopy(userDict)
    #commonMovieDict = {}
    movieCountDict = {}
    # initial movie count dict
    for i in range(1682):
        movieCountDict[i + 1] = 0
    commonViewDict = {}
    for key in userDict:
        user1 = userDict[key]
        findMovieCount(movieCountDict, user1)
        compareDict.pop(key, None)
        for key2 in compareDict:
            user2 = compareDict[key2]
            findCommonView(commonViewDict, user1, user2, key, key2)
            #findCommonMovie(commonMovieDict, user1, user2)
    return movieCountDict, commonViewDict

def problem1a(commonViewDict):
    countList = []
    for key in commonViewDict:
        countList.append(commonViewDict[key])
    countDict = {}
    
    for counter in countList:
        if counter in countDict:
            preCounter = countDict[counter]
            countDict[counter] = preCounter + 1
        else:
            countDict[counter] = 1
    #print countDict, '\n',len(countList), len(countDict)
    xList = []
    yList = []
    for key in countDict:
        xList.insert(0, key)
        yList.insert(0, countDict[key])
    print 'median value is: %s\nmean value is: %s' %(np.median(countList), np.mean(countList))
    plt.title('problem1a')
    plt.xlabel('number of movies reviewed in common')
    plt.ylabel('number of user pairs who have reviewed that many movies in common')
    plt.bar(xList, yList, 1)
    plt.show()
    plt.title('problem1a')
    plt.xlabel('number of movies reviewed in common')
    plt.ylabel('number of user pairs who have reviewed that many movies in common (log10 base)')
    plt.bar(xList, np.log10(yList), 1)
    plt.show()

def problem1b(movieDict):
    countList = [] 
    for key in movieDict:
        countList.append(movieDict[key])
    maxV = max(countList)
    minV = min(countList)
    xList = []
    yList = []
    maxList = []
    minList = []
    for w in sorted(movieDict, key = movieDict.get, reverse = False):
        xList.insert(0, w)
        yList.insert(0, movieDict[w])
        if movieDict[w] == maxV:
            maxList.append(w)
        if movieDict[w] == minV:
            minList.append(w)
    maxStr = ''.join(str(item)+' ' for item in maxList)
    minStr = ''.join(str(item)+' ' for item in minList)
    print 'max value is: %s, with movie id: %s\nmin value is: %s, with movie id: %s' %(str(maxV), maxStr, str(minV), minStr)
    x = [n for n in range(1, len(xList) + 1)]
    plt.title('problem1b')
    plt.xlabel('movieid')
    plt.ylabel('number of reviews')
    plt.plot(x, yList, '-')
    xLabels = []
    for i in range(len(xList)):
        if i % 100 == 0:
            xLabels.insert(i, str(xList[i]))
        else:
            xLabels.insert(i, '')
    plt.xticks(x, xLabels)
    plt.autoscale(tight=True)
    plt.show()
    plt.title('law of Zipf, log10 base')
    plt.xlabel('log10(movie id length)')
    plt.ylabel('log10(number of reviews)')
    plt.plot(np.log10(x), np.log10(yList), '-')
    plt.show()
    # use xticklabel matplotlib https://www.google.com/search?client=safari&rls=en&q=xticklabel&ie=UTF-8&oe=UTF-8#newwindow=1&q=xticklabel+matplotlib
    
def writeToJson(userDict, path):
    with open(path, 'w') as f:
        json.dump(userDict, f)

def readJson(path):
    with open(path, 'r') as f:
        try:
            userDict = json.load(f)
        except ValueError:
            userDict = {}
    return userDict

def main():
    uidList, itemIdList = readUData('./u.data')
    userDict = getUserDict(uidList, itemIdList)
    movieCountDict, commonViewDict = getCommonDict(userDict)
    problem1a(commonViewDict)
    problem1b(movieCountDict)



if __name__ == "__main__":
    sys.exit(main())
