# Starter code for uesr-based collaborative filtering
# Complete the function user_based_cf below. Do not change it arguments and return variables. 
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
        reader = csv.reader(csvfile, delimiter='\t')
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
    userDict = {}
    with open(path, 'r') as f:
        try:
            tempDict = json.load(f)
        except ValueError:
            print 'value error'
    for key in tempDict:
        userDict[int(key)] = tempDict[key]
    return userDict

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

def userHelper(userDict, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    userArray = copy.deepcopy(userDict[userid])
    distanceDict = {}
    userArray[1].pop(movieid - 1)
    #print userArray
    for key in userDict:
        if key == userid:
            continue
        compareArray = copy.deepcopy(userDict[key][1])
        compareArray.pop(movieid - 1)
        #print compareArray
        if distance == 0:
            distanceDict[key] = scipy.stats.pearsonr(userArray[1], compareArray)[0]
        else:
            distanceDict[key] = scipy.spatial.distance.cityblock(userArray[1], compareArray)
    #print distanceDict
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
        if iFlag == 0 and userDict[neighborId][1][movieid - 1] == 0:
            counter += 1
            continue
        neighborIdList.append(neighborId)
        index += 1
    #print counter, len(distanceDict)
    if counter == len(distanceDict):
        print 'special case: all other user have not rate this movie yet. case:', userid, movieid
        neighborIdList.append(userid)
    neighborRatingList = []
    for neighborId in neighborIdList:
        neighborRatingList.append(userDict[neighborId][1][movieid - 1])
    #print neighborIdList, neighborRatingList
    predictedRating = scipy.stats.mode(neighborRatingList)[0][0]
    trueRating = userDict[userid][1][movieid - 1]
    return trueRating, predictedRating

def user_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    uidList, itemIdList, ratingList = readUData(datafile)
    userDict = getUserDict(uidList, itemIdList, ratingList, numOfUsers, numOfItems)
    #path = './userDict.json'
    #writeToJson(userDict, path)
    #userDict = readJson(path)
    #print userDict
    return userHelper(userDict, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems)


def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682

    trueRating, predictedRating = user_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()
