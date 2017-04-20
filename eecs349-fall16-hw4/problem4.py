# EECS 349 Machine Learning
# HW4
# @Author: Liangji Wang

import csv #this would be for reading our text files
import sys
import numpy as np
import copy
import json
import random
import scipy.stats
import matplotlib.pyplot as plt
import user_cf
import item_cf
import math

# read double col file, in this hw are the wikitypo, wikitypoclean files
def readUData(path):
    list1 = []
    list2 = []
    list3 = []
    #list4 = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile,delimiter='\t')
        for row in reader:
            list1.insert(0, int(row[0]))
            list2.insert(0, int(row[1]))
            list3.insert(0, int(row[2]))
            #list4.insert(0, int(row[3]))
    return list1, list2, list3#, list4

def writeToJson(dataDict, path):
    with open(path, 'w') as f:
        json.dump(dataDict, f)

def readJson(path):
    dataDict = {}
    with open(path, 'r') as f:
        try:
            tempDict = json.load(f)
        except ValueError:
            print 'value error'
    for key in tempDict:
        dataDict[int(key)] = tempDict[key]
    return dataDict

def getSample(uidList, itemIdList, ratingList):
    tempList = []
    numToSelect = 100
    for i in range(len(uidList)):
        tempList.append(i)
    tempList = random.sample(tempList, numToSelect)
    sampleList = []
    for index in tempList:
        sampleList.append([uidList[index], itemIdList[index], ratingList[index]])
    return sampleList

def getSampleDict(uidList, itemIdList, ratingList):
    sampleDict = {}
    for i in range(50):
        sampleDict[i] = getSample(uidList, itemIdList, ratingList)
    return sampleDict

def createZeroList(n):
    return [0] * n

def getUserDict(uidList, itemIdList, ratingList, numOfUsers, numOfItems):
    # key is uid, value is a list of 2 lists: movie id list, rating list
    userDict = {}
    for i in range(1, numOfUsers + 1):
        # a tuple of two list, at 0 is movieID, at 1 is rating
        # np.zeros(numOfItems, dtype = int)
        userDict[i] = (createZeroList(numOfItems), createZeroList(numOfItems))
    for i in range(len(uidList)):
        uid = uidList[i]
        movieId = itemIdList[i]
        rating = ratingList[i]
        if uid in userDict:
            userDict[uid][0][movieId - 1] = 1
            userDict[uid][1][movieId - 1] = rating
        else:
            print 'error, uid not exist'
    return userDict

def getItemDict(uidList, itemIdList, ratingList, numOfUsers, numOfItems):
    # key is itemId, value is a list of 2 lists: uid list, rating list
    itemDict = {}
    for i in range(1, numOfItems + 1):
        # a tuple of two list, at 0 is uid, at 1 is rating
        # np.zeros(numOfItems, dtype = int)
        itemDict[i] = (createZeroList(numOfUsers), createZeroList(numOfUsers))
    for i in range(len(itemIdList)):
        uid = uidList[i]
        itemId = itemIdList[i]
        rating = ratingList[i]
        if itemId in itemDict:
            itemDict[itemId][0][uid - 1] = 1
            itemDict[itemId][1][uid - 1] = rating
        else:
            print 'error, itemId not exist'
    return itemDict

# userDict, userid, movieid, distance 0=pearson/1=manhattan, k, iFlag 0=non-0 rating/1=with 0 rating, numOfUsers, numOfItems
# userHelper(userDict, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems)
# def itemHelper(itemDict, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems)
# return trueRating, predictedRating
# simply pick k = 4 and i = 0
def result4C(sampleDict, itemDict, numOfUsers, numOfItems):
    pearsonDisResult = {}
    manhattanDisResult = {}
    errorP = {}
    errorM = {}
    for key in sampleDict:
        sampleList = sampleDict[key]
        errorPer = 0.0
        errorMan = 0.0
        counterP = 0
        counterM = 0
        for sample in sampleList:

            userid = sample[0]
            movieid = sample[1]
            rating = sample[2]
            # d=0, pearson
            trueRating, predictedRating = item_cf.itemHelper(itemDict, userid, movieid, 0, 4, 0, numOfUsers, numOfItems)
            errorPer += math.pow(trueRating - predictedRating, 2)
            if trueRating != predictedRating:
                counterP += 1
            # d=1, manhattan
            trueRating, predictedRating = item_cf.itemHelper(itemDict, userid, movieid, 1, 4, 0, numOfUsers, numOfItems)
            errorMan += math.pow(trueRating - predictedRating, 2)
            if trueRating != predictedRating:
                counterM += 1
            #print errorPer/counter, errorMan/counter
        pearsonDisResult[key] = errorPer/100
        manhattanDisResult[key] = errorMan/100
        errorP[key] = counterP/100.0
        errorM[key] = counterM/100.0
    writeToJson(pearsonDisResult, 'pearsonDisResult.json')
    writeToJson(manhattanDisResult, 'manhattanDisResult.json')
    #writeToJson(errorP, 'errorP.json')
    #writeToJson(errorM, 'errorM.json')

# simply pick d = 0 and k = 4
def result4D(sampleDict, userDict, numOfUsers, numOfItems):
    i0Result = {}
    i1Result = {}
    err0 = {}
    err1 = {}
    for key in sampleDict:
        sampleList = sampleDict[key]
        error0 = 0.0
        error1 = 0.0
        counter0 = 0.0
        counter1 = 0.0
        print 'sample:', key
        counter = 0
        for sample in sampleList:
            print counter, sample
            counter += 1
            userid = sample[0]
            movieid = sample[1]
            rating = sample[2]
            # d=0, pearson
            trueRating, predictedRating = user_cf.userHelper(userDict, userid, movieid, 0, 4, 0, numOfUsers, numOfItems)
            error0 += math.pow(trueRating - predictedRating, 2)
            if trueRating != predictedRating:
                counter0 += 1
            # d=1, manhattan
            trueRating, predictedRating = user_cf.userHelper(userDict, userid, movieid, 0, 4, 1, numOfUsers, numOfItems)
            error1 += math.pow(trueRating - predictedRating, 2)
            if trueRating != predictedRating:
                counter1 += 1
        i0Result[key] = error0/100
        i1Result[key] = error1/100
        err0[key] = counter0 / 100
        err1[key] = counter1 / 100
    writeToJson(i0Result, 'i0Result.json')
    writeToJson(i1Result, 'i1Result.json')
    #writeToJson(err0, 'err0.json')
    #writeToJson(err1, 'err1.json')

def analysis(path1, path2, title):
    print path1, path2
    mse1Result = readJson(path1)
    mse2Result = readJson(path2)
    mse1List = []
    mse2List = []
    mmse1 = 0
    mmse2 = 0
    for i in range(50):
        mse1List.insert(0, mse1Result[i])
        mse2List.insert(0, mse2Result[i])
        mmse1 += mse1Result[i]
        mmse2 += mse2Result[i]
    print 'exp:', str(title)
    #print 'mse1 normal test:', scipy.stats.normaltest(mse1List)
    #print 'mse2 normal test:', scipy.stats.normaltest(mse2List)
    #print 'wilcoxon-test', scipy.stats.wilcoxon(mse1List, mse2List)
    print 'p-test:', scipy.stats.ttest_rel(mse1List, mse2List)
    print 'mmse1 is %s, mmse2 is %s' %(mmse1 / 50, mmse2 / 50)
    # plot box below
    labels = [path1, path2]
    plt.boxplot([mse1List, mse2List], labels=labels, showmeans=True)
    plt.ylabel('mse value')
    plt.title(str(title))
    plt.show()

def result4E(sampleDict, userDict, numOfUsers, numOfItems):
    kList = [1, 2, 4, 8, 16, 32]
    distance = 0 #pearson is better
    iFlag = 0
    # dict of dict: first key is sample no., then key is k value, then store the value of true - predict
    kResult = {}
    for key in sampleDict:
        kResult[key] = {}
        for k in kList:
            kResult[key][k] = []
    for key in sampleDict:
        sampleList = sampleDict[key]
        print 'sample:', key
        counter = 0
        for sample in sampleList:
            print counter, sample
            counter += 1
            userid = sample[0]
            movieid = sample[1]

            for k in kList:
                trueRating, predictedRating = user_cf.userHelper(userDict, userid, movieid, distance, k, iFlag, numOfUsers,
                                                                 numOfItems)
                kResult[key][k].append(trueRating - predictedRating)
    writeToJson(kResult, 'kResult.json')

def splitKResult(path):
    kList = [1, 2, 4, 8, 16, 32]
    kResult = readJson(path)
    kDict = {}
    for k in kList:
        kDict[k] = {}
    for i in range(50):
        tempDict = kResult[i]
        for key in tempDict:
            diffList = tempDict[key]
            mse = 0.0
            for item in diffList:
                mse += math.pow(item, 2)
            kDict[int(key)][i] = mse / len(diffList)
    for k in kList:
        mseDict = kDict[k]
        mmse = 0.0
        for sampleNo in mseDict:
            mmse += mseDict[sampleNo]
        print 'mmse for k = %s is %s' %(k, mmse/50)
        writeToJson(mseDict, 'k'+str(k)+'.json')

def plotK(path1, path2, path3, path4, path5, path6):
    mse1Result = readJson(path1)
    mse2Result = readJson(path2)
    mse3Result = readJson(path3)
    mse4Result = readJson(path4)
    mse5Result = readJson(path5)
    mse6Result = readJson(path6)
    mse1List = []
    mse2List = []
    mse3List = []
    mse4List = []
    mse5List = []
    mse6List = []
    for i in range(50):
        mse1List.insert(0, mse1Result[i])
        mse2List.insert(0, mse2Result[i])
        mse3List.insert(0, mse3Result[i])
        mse4List.insert(0, mse4Result[i])
        mse5List.insert(0, mse5Result[i])
        mse6List.insert(0, mse6Result[i])
    # plot box below
    labels = [path1, path2, path3, path4, path5, path6]
    #labels = ['1', '2', '4', '8', '16', '32']
    plt.boxplot([mse1List, mse2List, mse3List,mse4List,mse5List,mse6List], labels=labels, showmeans=True)
    plt.ylabel('mse value')
    plt.title('k = 1, 2, 4, 8, 16, 32')
    plt.show()

def result4F(sampleDict, userDict, itemDict, numOfUsers, numOfItems):
    userResult = {}
    itemResult = {}
    iFlag = 0
    kValue = 32
    distance = 0
    for key in sampleDict:
        sampleList = sampleDict[key]
        userMse = 0.0
        itemMse = 0.0
        index = 0
        for sample in sampleList:
            print key, index, sample
            index += 1
            userid = sample[0]
            movieid = sample[1]
            rating = sample[2]
            # user base
            trueRating, predictedRating = user_cf.userHelper(userDict, userid, movieid, distance, kValue, iFlag, numOfUsers, numOfItems)
            userMse += math.pow(trueRating - predictedRating, 2)
            # item base
            trueRating, predictedRating = item_cf.itemHelper(itemDict, userid, movieid, distance, kValue, iFlag, numOfUsers, numOfItems)
            itemMse += math.pow(trueRating - predictedRating, 2)
            #print errorPer/counter, errorMan/counter
        userResult[key] = userMse/100
        itemResult[key] = itemMse/100
        #print userResult, itemResult
    writeToJson(userResult, 'userResult.json')
    writeToJson(itemResult, 'itemResult.json')

def main():
    numOfUsers = 943
    numOfItems = 1682
    kList = [1, 2, 4, 8, 16, 32]
    #uidList, itemIdList, ratingList = readUData('./u.data')
    #sampleDict = getSampleDict(uidList, itemIdList, ratingList)
    #userDict = getUserDict(uidList, itemIdList, ratingList, numOfUsers, numOfItems)
    #itemDict = getItemDict(uidList, itemIdList, ratingList, numOfUsers, numOfItems)
    #writeToJson(sampleDict, 'sampleDict.json')
    #writeToJson(userDict, 'userDict.json')
    #writeToJson(itemDict, 'itemDict.json')
    sampleDict = readJson('sampleDict.json')
    userDict = readJson('userDict.json')
    itemDict = readJson('itemDict.json')
    #result4C(sampleDict, itemDict, numOfUsers, numOfItems)
    #result4D(sampleDict, userDict, numOfUsers, numOfItems)
    #analysis('pearsonDisResult.json', 'manhattanDisResult.json', 'problem 4c')
    #analysis('errorP.json', 'errorM.json')
    #analysis('i0Result.json', 'i1Result.json', 'problem 4d')
    #analysis('err0.json', 'err1.json')
    #result4E(sampleDict, userDict, numOfUsers, numOfItems)
    #splitKResult('kResult.json')
    #result4F(sampleDict, userDict, itemDict, numOfUsers, numOfItems)
    plotK('k1.json', 'k2.json', 'k4.json', 'k8.json', 'k16.json', 'k32.json')
    #for k in kList:
    #    if k == 32:
    #        continue
    #    analysis('k32.json', 'k'+str(k)+'.json', 'k32 vs k'+str(k))
    #result4F(sampleDict, userDict, itemDict, numOfUsers, numOfItems)
    #analysis('userResult.json', 'itemResult.json', 'problem4f user vs item')


if __name__ == "__main__":
    sys.exit(main())
