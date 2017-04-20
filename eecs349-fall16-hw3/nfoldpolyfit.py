#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
import copy

# lib for plotForExp function
from mpl_toolkits.mplot3d import axes3d
import math
from matplotlib import cm

#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Liangji Wang
#

# the nFold cross validation helper function
# return the indexDict
def getFolds(X, n, verbose):
	foldIndexDict = {}
	indexList = [i for i in xrange(len(X))]
	random.shuffle(indexList)
	foldNo = 0
	tempList = []
	flag = False
	if len(X) % n != 0:
		flag = True
	for i in range(len(indexList)):
		tempList.append(indexList[i])
		if (i+1) % (len(X) / n) == 0:
			foldIndexDict[foldNo] = tempList
			tempList = []
			foldNo += 1
	# if len(X) % n != 0, add the rest elements to the last fold list
	foldIndexDict[foldNo - 1] = foldIndexDict[foldNo - 1] + tempList
	#print foldIndexDict
	return foldIndexDict

# get X and Y List by using indexList
def getXY(X, Y, indexList): 
	x = []
	y = []
	for index in indexList:
		x.insert(0, X[index])
		y.insert(0, Y[index])
	return x, y

# calculate MSE based on existing value y and weight function value f(x)
def calMSE(y_test, y_exp):
	mse = 0.0
	for i in range(len(y_test)):
		mse += pow(y_test[i] - y_exp[i], 2)
	return mse / float(len(y_test))

# do the poly fit for n fold
def nfoldpolyfit(X, Y, maxK, n, verbose):
	minMSE = sys.maxint
	# fold the data set
	foldIndexDict = getFolds(X, n, verbose)
	mmseForKDict = {}
	bestFuncList = []
	for k in range(maxK + 1):
		#print 'k is:', k
		mmseForK = 0.0
		for i in range(n):
			#print 'test fold No.', i
			#print i 
			indexList_Train = []
			for key in foldIndexDict:
				#print 'key is:', key, type(key)
				#print 'i is:', i, type(i)
				if n == 1:
					print '******************************\nspecial case with only 1 fold,\nthus training and testing list will be same\n******************************'
					indexList_Train += foldIndexDict[key]
					break
				if key != i:
					indexList_Train += foldIndexDict[key]
			#print 'indexList_Train is:', indexList_Train
			#print 'get train x, y'
			x_train, y_train = getXY(X, Y, indexList_Train)
			#print x_train, y_train
			indexList_Test = foldIndexDict[i]
			#print 'get test x, y'
			x_test, y_test = getXY(X, Y, indexList_Test)
			#print x_test, y_test
			factors = np.polyfit(x_train, y_train, k)
			#print 'factors is:', factors
			y_exp = np.polyval(factors, x_test)
			#print 'y_exp is:',y_exp
			mse = calMSE(y_exp, y_test)
			#print 'mse is:', mse
			mmseForK += mse
		mmseForK /= float(n)
		mmseForKDict[k] = mmseForK
		if minMSE > mmseForK:
			minMSE = mmseForK
			del bestFuncList[:]
			bestFuncList = [k, minMSE, factors]
	#print mmseForKDict
	if verbose:
		plotProblem1a(mmseForKDict, bestFuncList, n, X, Y)
	return mmseForKDict, bestFuncList

# plot helper function for my exp
# show a 3D graph 
# replace extreme value to the largest value in the graph metric
def plotForExp(mmseForFoldDict):
	foldValueList = []
	kValueList = []
	mMseList = []
	for foldV in mmseForFoldDict:
		for kV in mmseForFoldDict[foldV]:
			foldValueList.insert(0, foldV)
			kValueList.insert(0, kV)
			mmse = mmseForFoldDict[foldV][kV]
			#deal extreme value
			if mmse > 1.5:
				mmse = 1.5
			mMseList.insert(0, mmse)
	fig = plt.figure()
	sub0 = fig.add_subplot(111, projection = '3d')
	sub0.set_title('plot for Exp')
	sub0.set_xlabel('fold value')
	sub0.set_ylabel('k value')
	sub0.set_zlabel('Mean MSE value')
	sub0.set_xlim(0, 15)
	sub0.set_ylim(0, 10)
	sub0.set_zlim(0, 1.5)
	sub0.view_init(elev=12, azim=40)
	sub0.dist=12 
	sub0.plot_trisurf(foldValueList, kValueList, mMseList, cmap=cm.jet, linewidth=0.1)
	plt.show()

# exp function to do the exp and help decide the best k value
def doExp(X, Y, verbose):
	#foldValueList = [2, 3, 5, 6, 10, 15]
	# get fold value list from 2 to 15
	foldValueList = [i for i in range(2, 16)]
	mmseForFoldDict = {}
	lowestMSE = sys.maxint
	lowestFoldV = 0
	lowestKV = 0
	# call nfoldpolyfit and get all MMSE for all k and all fold
	for foldV in foldValueList:
		mmseForFoldDict[foldV], bestFuncList= nfoldpolyfit(X, Y, 9, foldV, verbose)
	# now analysis the dataDict and print the lowest mmse for each fold and in all exp 
	for foldV in mmseForFoldDict:
		minMSE = sys.maxint
		minkV = 0
		print 'foldV:', foldV
		for kValue in mmseForFoldDict[foldV]:
			mmse = mmseForFoldDict[foldV][kValue]
			print 'kValue:', kValue, 'mmse:', mmse
			if mmse < minMSE:
				minMSE = mmse
				minkV = kValue 
			if mmse < lowestMSE:
				lowestMSE = mmse
				lowestKV = kValue
				lowestFoldV = foldV
		print 'lowest mean mse for foldV %s with k = %s is: %s' %(foldV, minkV, minMSE)
	print 'lowest mean mse in this exp is: %s when k = %s and folds = %s ' %(lowestMSE, lowestKV, lowestFoldV)
	plotForExp(mmseForFoldDict)

# plot function for problem 1 a
def plotProblem1a(mmseForKDict, bestFuncList, foldNo, X, Y):
	# process the data
	kList = []
	mMseList = []
	# data dict processing
	for kV in mmseForKDict:
		kList.insert(0, kV)
		mMseList.insert(0, mmseForKDict[kV])
	funcK = bestFuncList[0]
	funcMMSE = bestFuncList[1]
	funcFactorList = bestFuncList[2]
	# do plot
	fig = plt.figure()
	sub0 = fig.add_subplot(211)
	sub0.set_title('problem1a with fold = ' + str(foldNo) + '\nMean MSE for each K value')
	sub0.set_xlabel('K value')
	sub0.set_ylabel('Mean MSE')
	sub0.plot(kList, mMseList)
	sub1 = fig.add_subplot(212)
	sub1.set_title('best function with K = ' + str(funcK) + ' and Mean MSE is ' + str(funcMMSE))
	sub1.set_xlabel('X')
	sub1.set_ylabel('Y')
	newX = copy.deepcopy(X)
	newX.sort()
	curveX = np.linspace(newX[0], newX[-1], 200)
	curveY = []
	for i in range(len(curveX)):
		curveY.insert(i, np.polyval(funcFactorList, curveX[i]))
	sub1.plot(X, Y, '.', curveX, curveY, '-')
	plt.show()

def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	verbose = int(sys.argv[4])
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	mmseForKDict, bestFuncList = nfoldpolyfit(X, Y, maxK, nFolds, verbose)
	funcK = bestFuncList[0]
	funcMMSE = bestFuncList[1]
	funcFactorList = bestFuncList[2]
	print 'best function weight is: %s, at k = %s' %(funcFactorList, funcK)

	#doExp(X, Y, verbose)

if __name__ == "__main__":
	main()

