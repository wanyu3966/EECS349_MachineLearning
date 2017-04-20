import pickle
import sklearn
from mnist import load_mnist
from sklearn import svm # this is an example of using SVM
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier # use multi-layer perceptron algorithm (neural network)
from sklearn.model_selection import KFold

import random
import sys

from sklearn.metrics import confusion_matrix

def dimension_reduce(vector):
    #dimension reduction using pca
    pca = PCA(n_components=28)
    pca.fit(vector)
    feature_vector = pca.explained_variance_
    return feature_vector

def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]

def build_classifier(images, labels, actFn='tanh', sol='adam', batch=5, epoch=100, hiddenNodeNum=200, learnRate=0.001):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    print 'train data size: %s, actFn: %s, solver: %s, batch size: %s, epoch size: %s, hiddenNodeNum: %s\n' %(len(labels), actFn, sol, batch, epoch, hiddenNodeNum)
    classifier = MLPClassifier(activation=actFn, solver=sol, batch_size=batch, max_iter=epoch, hidden_layer_sizes=(hiddenNodeNum, 10), learning_rate_init=learnRate)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    # pickle.dump(training_set, open('training_set.p', 'w'))
    # pickle.dump(training_labels, open('training_labels.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    print 'test data size: %s' %(len(images))
    return classifier.predict(images)

def error_measure(predicted, actual):
    error = np.count_nonzero(abs(predicted - actual))/float(len(predicted))
    print error
    return error

def dataSplit(images, labels, numDataToUse=60000, numCross=10000 ,numTest=20000):
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

def getCrossSet(crossData, crossLabel, foldNum=10):
    kf = KFold(n_splits=foldNum)
    trainIndexList = []
    testIndexList = []
    i = 0
    for train, test, in kf.split(crossData):
        trainIndexList.insert(i, train)
        testIndexList.insert(i, test)
        i += 1
    trainSetList = []
    trainLabelList = []
    testSetList = []
    testLabelList = []
    for i in range(foldNum):
        trainSetList.append([])
        trainLabelList.append([])
        testSetList.append([])
        testLabelList.append([])
        for index in trainIndexList[i]:
            trainSetList[i].append(crossData[index])
            trainLabelList[i].append(crossLabel[index])
        for index in testIndexList[i]:
            testSetList[i].append(crossData[index])
            testLabelList[i].append(crossLabel[index])
    return trainSetList, trainLabelList, testSetList, testLabelList

def expForPara(trainSetList, trainLabelList, testSetList, testLabelList, foldNum=10):
    # for MLPClassifier
    # activation : {'identity', 'logistic', 'tanh', 'relu'}
    # solver : {'lbfgs', 'sgd', 'adam'}
    # build_classifier(images, labels, actFn='relu', sol='sgd', batch=10, epoch=200, hiddenNodeNum=50, learnRate=0.001)
    actList = ['identity', 'logistic', 'tanh', 'relu']
    solvList = ['lbfgs', 'sgd', 'adam']
    batchList=[5, 10, 20, 40, 80, 160, 320, 640]
    epochList=[10, 50, 100, 500, 1000, 2000, 5000]
    hiddenNodeList=[10, 20, 50, 100, 200, 500]

    errorList = []
    for item in actList:
        tempErr=[]
        for i in range(foldNum):
            classifier = build_classifier(trainSetList[i], trainLabelList[i], actFn=item)
            predicted = classify(testSetList[i], classifier)
            tempErr.append(error_measure(predicted, testLabelList[i]))
        meanErr = sum(tempErr)/float(foldNum)
        print item, tempErr, 'mean error rate:', meanErr
        errorList.append(meanErr)
    print actList, errorList
    return errorList

def flatToDim(flatData, n=28):
    dimData = np.ndarray(shape=(n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dimData[i][j] = flatData[i*n + j]
    return dimData

def plotFlatImage(label, flatData, n=28, saveName=None):
    plt.imshow(flatToDim(flatData), cmap='gray')
    plt.title('handwrite image of the digit' + str(label))
    if saveName != None:
        plt.savefig(saveName)
    else:
        plt.show()

if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(10), path='.')
    # preprocessing
    images = preprocess(images)
    # print images[0]
    # plotFlatImage(labels[0], images[0], saveName=('c1MissData' + str('flag') + '.png'))

    for i in range(10):
        crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels, numDataToUse=50000, numCross=10000 ,numTest=20000)
        classifier = build_classifier(trainData, trainLabel, actFn='tanh', sol='adam', batch=5, epoch=200, hiddenNodeNum=200, learnRate=0.001)
        predicted = classify(testData, classifier)
        print error_measure(predicted, testLabel)
    # cross validation start here


    # crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels, numDataToUse=50000, numCross=10000 ,numTest=20000)

    # classifier = build_classifier(trainData, trainLabel)
    # save_classifier(classifier, trainData, trainLabel)
    # classifier = pickle.load(open('classifier_1.p'))
    # print 'finish build'
    # predicted = classify(testData, classifier)
    # print 'finish classify'
    # flag = 0
    # for i in range(len(predicted)):
    #     if flag > 20:
    #         break
    #     preLabel = predicted[i]
    #     trueLabel = testLabel[i]
    #     if trueLabel == 2 and preLabel != trueLabel:
    #         plotFlatImage(trueLabel, testData[i], saveName=('c1MissData' + str(flag) + 'predictedAs' + str(preLabel) + '.png'))
    #         flag += 1
    # print error_measure(predicted, testLabel)
    # print confusion_matrix(testLabel, predicted)


    # trainSetList, trainLabelList, testSetList, testLabelList = getCrossSet(crossData, crossLabel)
    
    # expForPara(trainSetList, trainLabelList, testSetList, testLabelList)




    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    # training_set = images[0:1000]
    # training_labels = labels[0:1000]
    # testing_set = images[-100:]
    # testing_labels = labels[-100:]

    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    # classifier = build_classifier(training_set, training_labels)
    # save_classifier(classifier, training_set, training_labels)
    # classifier = pickle.load(open('classifier_1.p'))
    # predicted = classify(testing_set, classifier)
    # print error_measure(predicted, testing_labels)
