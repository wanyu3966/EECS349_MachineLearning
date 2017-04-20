import pickle
import sklearn
from sklearn import svm # this is an example of using SVM
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from sklearn.decomposition import PCA
import random
from sklearn.metrics import confusion_matrix

def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]

def dimension_reduce(vector):
    #dimension reduction using pca
    pca = PCA(n_components=28)
    pca.fit(vector)
    feature_vector = pca.explained_variance_
    return feature_vector

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

def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = svm.SVC(C=100, kernel='rbf', gamma=0.01)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_2.p', 'w'))
    pickle.dump(training_set, open('training_set_2.p', 'w'))
    pickle.dump(training_labels, open('training_labels_2.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(10), path='.')

    # preprocessing
    images = preprocess(images)
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    crossData, crossLabel, training_set, training_labels, testing_set, testing_labels = dataSplit(images, labels)

    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    classifier = build_classifier(training_set, training_labels)
    save_classifier(classifier, training_set, training_labels)
    classifier = pickle.load(open('classifier_2.p'))
    predicted = classify(testing_set, classifier)
    print error_measure(predicted, testing_labels)
    print confusion_matrix(testing_labels, predicted)
