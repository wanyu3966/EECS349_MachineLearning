# Starter code for item-based collaborative filtering
# Complete the function item_based_cf below. Do not change its name, arguments and return variables. 
# Do not change main() function, 

# import modules you need here.
import sys
import csv
import scipy.stats
import scipy.spatial.distance
import numpy as np
import json
import copy

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

def writeToJson(userDict, path):
    with open(path, 'w') as f:
        json.dump(userDict, f)

def readJson(path):
    itemDict = {}
    with open(path, 'r') as f:
        try:
            tempDict = json.load(f)
        except ValueError:
            print 'value error'
    for key in tempDict:
        itemDict[int(key)] = tempDict[key]
    return itemDict

def createZeroList(n):
    return [0] * n

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

def itemHelper(itemDict, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    itemArray = copy.deepcopy(itemDict[movieid])
    distanceDict = {}
    itemArray[1].pop(userid - 1)

    for key in itemDict:
        if key == movieid:
            continue
        compareArray = copy.deepcopy(itemDict[key][1])
        compareArray.pop(userid - 1)
        if distance == 0:
            distanceDict[key] = scipy.stats.pearsonr(itemArray[1], compareArray)[0]
        else:
            distanceDict[key] = scipy.spatial.distance.cityblock(itemArray[1], compareArray)
    neighborIdList = []
    index = 0
    counter = 0
    # flag for asscending/descending sorts, True for descending
    if distance == 0:
        flag = True
    else:
        flag = False
    #print distanceDict
    for neighborId in sorted(distanceDict, key = distanceDict.get, reverse = flag):
        if index >= k:            
            break
        if iFlag == 0 and itemDict[neighborId][1][userid - 1] == 0:
            counter += 1
            continue
        neighborIdList.append(neighborId)
        index += 1
    if counter == len(distanceDict):
        print 'special case:', userid, movieid
        neighborIdList.append(movieid)
    neighborRatingList = []
    for neighborId in neighborIdList:
        neighborRatingList.append(itemDict[neighborId][1][userid - 1])
    predictedRating = scipy.stats.mode(neighborRatingList)[0][0]
    trueRating = itemDict[movieid][1][userid - 1]

    return trueRating, predictedRating

def item_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    uidList, itemIdList, ratingList = readUData(datafile)
    itemDict = getItemDict(uidList, itemIdList, ratingList, numOfUsers, numOfItems)
    #path = './itemDict.json'
    #writeToJson(itemDict, path)
    #itemDict = readJson(path)
    #print itemDict
    return itemHelper(itemDict, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems)


def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682

    trueRating, predictedRating = item_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()
