#experiment code for spam filter assignment in EECS349 Machine Learning
#Author: Liangji Wang

import sys
import numpy as np
import os
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import json
import random
import copy

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
            tempDict = {}
    for key in tempDict:
        dataDict[key] = tempDict[key]
    return dataDict

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

#def getDict(spam_directory, ham_directory, trainDirList, fileDict):
def getDict(trainDirList, fileDict):
	#Making the dictionary. 
	spam = []
	ham = []
	for trainDir in trainDirList:
		fileList = [f for f in os.listdir(trainDir) if os.path.isfile(os.path.join(trainDir, f))]
		for email in fileList:
			if fileDict[email]:
				spam.append(trainDir + email)
			else:
				ham.append(trainDir + email)
	
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
	words = {}

	helpSet = set()
	spamCounter = 0
	hamCounter = 0
	for s in spam:
		spamCounter += 1
		for word in parse(open(s)):
			helpSet.add(word)
			if word not in words:
				words[word] = {'spam': 0, 'ham': 0}
		for word in helpSet:
			temp = words[word]['spam']
			words[word]['spam'] = temp + 1
		helpSet.clear()
			
	for h in ham:
		hamCounter += 1
		for word in parse(open(h)):
			helpSet.add(word)
			if word not in words:
				words[word] = {'spam': 0, 'ham': 0}
		for word in helpSet:
			temp = words[word]['ham']
			words[word]['ham'] = temp + 1
		helpSet.clear()

	spamCounter += 1
	hamCounter += 1

	for word in words:
		temp = words[word]['spam']
		if temp == 0:
			#print 'special case: word %s exist in ham but not spam, to avoid log(0), set its spam probability to 1/(spamCounter+1)' %(word)
			words[word]['spam'] = 1.0 / float(spamCounter)
		else:
			words[word]['spam'] = float(temp) / float(spamCounter)
		temp = words[word]['ham']
		if temp == 0:
			#print 'special case: word %s exist in spam but not ham, to avoid log(0), set it ham probability to 1/(hamCounter+1)' %(word)
			words[word]['ham'] = 1.0 / float(hamCounter)
		else:
			words[word]['ham'] = float(temp) / float(hamCounter)
	# check
	print 'dictionary information:\nspam file Counter: %s, ham file Counter: %s, dictLength: %s, spam_prior_probability: %s' %(spamCounter, hamCounter, len(words), spam_prior_probability)
	
	return words, spam_prior_probability

def is_spam(content, dictionary, spam_prior_probability, priorFlag = False):
	# use prior_probability if flag
	if priorFlag:
		if spam_prior_probability >= .5:
			return True
		return False
	# Vnb = argMax(log(prior_probability) + sum(log(P(ai|vj))))
	# use the log to help replace multiple with addition
	spamV = []
	hamV = []
	spamV.append(spam_prior_probability)
	hamV.append(1-spam_prior_probability)
	for word in content:
		if word in dictionary:
			spamV.append(dictionary[word]['spam'])
			hamV.append(dictionary[word]['ham'])
	#print sum(np.log10(spamV)), sum(np.log10(hamV))
	if sum(np.log10(spamV)) >= sum(np.log10(hamV)):
		return True
	return False

def spamTest(testDir, dictionary, spam_prior_probability, fileDict):
	errRateForNaive = 0.0;
	errRateForPrior = 0.0
	mail = [f for f in os.listdir(testDir) if os.path.isfile(os.path.join(testDir, f))]
	for m in mail:
		content = parse(open(testDir + m))
		spamForNaive = is_spam(content, dictionary, spam_prior_probability, False)
		spamForPrior = is_spam(content, dictionary, spam_prior_probability, True)
		if m not in fileDict:
			print 'file %s is not in fileDict, NO RECORD FOUND, set this case as error' %(m)
			errRateForNaive += 1
			errRateForPrior += 1
		else:
			if spamForNaive != fileDict[m]:
				errRateForNaive += 1
			if spamForPrior != fileDict[m]:
				errRateForPrior += 1
	errRateForNaive /= len(mail)
	errRateForPrior /= len(mail)
	print 'Test information:\nerror rate for Naive: %s, error rate for prior_probability: %s'%(errRateForNaive, errRateForPrior)
	return errRateForNaive, errRateForPrior

def getFileDict(spam_directory, ham_directory):
	fileDict = {}
	#Making the dictionary. 
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	
	for s in spam:
		if s in fileDict:
			print 'error: same file name'
		fileDict[s] = True
	for h in ham:
		if h in fileDict:
			print 'error: same file name'
		fileDict[h] = False
	return fileDict

# split data for cross validation
def dataSplit(spam_directory, ham_directory, foldNum, spamRate):
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	spamNum = len(spam)
	total = int(spamNum/spamRate)
	if len(ham) + spamNum < total:
		total = int(len(ham)/(1-spamRate))
		spamNum = total - len(ham)
	hamNum = total - spamNum
	usedFileDict = {}
	# random pick from spam and ham with spamRate
	random.shuffle(spam)
	random.shuffle(ham)
	randomList = []
	for fileName in spam[:spamNum]:
		randomList.append(spam_directory + fileName)
	for fileName in ham[:hamNum]:
		randomList.append(ham_directory + fileName)
	random.shuffle(randomList)
	numPerDir = total/foldNum
	for index in range(foldNum):
		if not os.path.exists(str(index)):
			os.mkdir(str(index))
		else:
			deleteList = [f for f in os.listdir(str(index)) if os.path.isfile(os.path.join(str(index), f))]
			for item in deleteList:
				#print item
				os.remove(str(index) + '/' + item)
		for count in range(numPerDir):
			shutil.copy(randomList.pop(), str(index))
		
	print 'total file can be split: %s, spam rate: %s, total splited file count: %s, total splited spam file: %s, total splited ham file: %s, folds: %s, file per Dir: %s' %(len(spam)+len(ham), spamRate, total, spamNum, hamNum, foldNum, numPerDir)
	
def getResult(naiveErrList, priorErrList):
	print 'Result:\nnaive error normal test: %s\nprior probability error normal test: %s\nmean of naive error rate: %s, mean of prior probability error rate: %s\nstudent t-test(paired): %s\nwilcoxon-test: %s' %(scipy.stats.normaltest(naiveErrList), scipy.stats.normaltest(priorErrList), np.mean(naiveErrList), np.mean(priorErrList), scipy.stats.ttest_rel(naiveErrList, priorErrList), scipy.stats.wilcoxon(naiveErrList, priorErrList))
	labels = ['naive error', 'prior_probability error']
	plt.boxplot([naiveErrList, priorErrList], labels = labels, showmeans = True)
	plt.ylabel('error rate')
	plt.title('exp result')
	plt.show()
	pass

if __name__ == "__main__":
	training_spam_directory = 'spam/'
	training_ham_directory = 'ham/'
	# create a dict to store wheter this file is a spam file or ham
	fileDict = getFileDict(training_spam_directory, training_ham_directory)
	# store to this dict to json file
	# writeToJson(fileDict, 'fileDict.json')
	# read json file to dict
	#fileDict = readJson('fileDict.json')

	# exp set up, parameter setting
	foldNum = 10
	spamRate = .57

	# do data split here
	dataSplit(training_spam_directory, training_ham_directory, foldNum, spamRate)

	# exp starts here
	expDirList = []
	naiveErrResult = []
	priorErrResult = []
	for i in range(foldNum):
		expDirList.insert(i, str(i) + '/')
	for i in range(foldNum):
		print 'test case: %s' %(expDirList[i])
		test_mail_directory = expDirList[i]
		trainDirList = copy.deepcopy(expDirList)
		trainDirList.pop(i)
		#print 'remove case %s from trainDirList' %(trainDirList.pop(i))
		dictionary, spam_prior_probability = getDict(trainDirList, fileDict)
		naiveErr, priorErr = spamTest(test_mail_directory, dictionary, spam_prior_probability, fileDict)
		naiveErrResult.insert(i, naiveErr)
		priorErrResult.insert(i, priorErr)
	getResult(naiveErrResult, priorErrResult)



