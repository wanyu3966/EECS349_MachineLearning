# import packages (including scikit-learn packages)
import sklearn
import random
import csv
import sys
from mnist import load_mnist
import numpy as np
from sklearn import svm # this is an example of using SVM
from sklearn.ensemble import AdaBoostClassifier # Use this function for adaboosting
from sklearn.metrics import confusion_matrix

def preprocess(images):
    #this function is suggested to help build your classifier.
    #You might want to do something with the images before
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]

def classify(images, classifier):
    #runs the classifier on a set of images.
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

def dataSplit(images, labels, numDataToUse=50000, numCross=10000 ,numTest=20000):
    imagesDict = {}
    for index in range(len(labels)):
        if labels[index] not in imagesDict:
            imagesDict[labels[index]] = []
        imagesDict[labels[index]].append(images[index])
    keyLen = len(imagesDict)
    crossData = []
    crossLabel = []
    trainData = []
    trainLabel = []
    testData = []
    testLabel = []
    crossLenEachNum = numCross / keyLen
    testLenEachNum = numTest / keyLen
    tempCrossData = []
    tempCrossLabel = []

    for key in imagesDict:
        singleNumList = imagesDict[key]
        random.shuffle(singleNumList)
        testData += singleNumList[0:testLenEachNum]
        testLabel += [key] * testLenEachNum
        trainData += singleNumList[-(len(singleNumList) - testLenEachNum):]
        trainLabel += [key] * (len(singleNumList) - testLenEachNum)
        tempCrossData += singleNumList[-(crossLenEachNum):]
        tempCrossLabel += [key] * crossLenEachNum
        # print len(testData), len(testLabel), len(trainData), len(trainLabel), len(tempCrossData), len(tempCrossLabel)

    indexList = range(len(tempCrossData))
    random.shuffle(indexList)
    for i in indexList:
        crossData.append(tempCrossData[i])
        crossLabel.append(tempCrossLabel[i])
    if len(trainData) > (numDataToUse - numTest):
        indexList = range(len(trainData))
        random.shuffle(indexList)
        tempData = []
        tempLabel = []
        flag = 0
        for i in indexList:
            if flag == (numDataToUse - numTest):
                return crossData, crossLabel, tempData, tempLabel, testData, testLabel
            tempData.append(trainData[i])
            tempLabel.append(trainLabel[i])
            flag += 1

    return crossData, crossLabel, trainData, trainLabel, testData, testLabel

def write_csv_file(file_name, data):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')

        for row in data:
            writer.writerow(row)


def write_csv_file_row(file_name, data):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        writer.writerow(data)

def boosting_A(training_set, training_labels, testing_set, testing_labels):
    '''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		  (NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)
	
	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels 
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''

    # Build boosting algorithm for question 6-A

    classifier = AdaBoostClassifier()

    classifier.fit(training_set, training_labels)

    predicted_labels = classify(testing_set, classifier)

    confusion_matrix_A = confusion_matrix(testing_labels, predicted_labels)

    return predicted_labels, confusion_matrix_A

def boosting_B(training_set, training_labels, testing_set, testing_labels):
    '''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		(NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)
	
	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels 
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''
    # Build boosting algorithm for question 6-B
    weak_classifier = svm.SVC(C=100, kernel='rbf', gamma=0.01, probability=True)

    classifier = AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=10)

    print len(training_set)

    classifier.fit(training_set, training_labels)

    predicted_labels = classify(testing_set, classifier)

    confusion_matrix_B = confusion_matrix(testing_labels, predicted_labels)

    return predicted_labels, confusion_matrix_B

def main():
    """
    This function runs boosting_A() and boosting_B() for problem 7.
    Load data set and perform adaboosting using boosting_A() and boosting_B()
    """
    # Code for loading data
    images, labels = load_mnist(digits=range(10), path='.')
    # preprocessing
    images = preprocess(images)

    crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels)

    predicted_labels_A, confusion_matrix_A = boosting_A(trainData, trainLabel, testData, testLabel)

    error_rate = error_measure(predicted_labels_A, testLabel)

    #write_csv_file_row('boost_A_predicted_labels.csv', predicted_labels_A)
    #write_csv_file('boost_A_confusion_matrix.csv', confusion_matrix_A)
    print "A:"
    print "Predicted Labels:", predicted_labels_A
    print "Confusion matirx:", confusion_matrix_A
    print "Error Rate:", error_rate
    print

    predicted_labels_B, confusion_matrix_B = boosting_B(trainData, trainLabel, testData, testLabel)
    error_rate = error_measure(predicted_labels_B, testLabel)

    #write_csv_file_row('boost_B_predicted_labels.csv', predicted_labels_B)
    #write_csv_file('boost_B_confusion_matrix.csv', confusion_matrix_B)
    print "B:"
    print "Predicted Labels:", predicted_labels_B
    print "Confusion matirx:", confusion_matrix_B
    print "Error Rate:", error_rate

if __name__ == '__main__':
    main()
