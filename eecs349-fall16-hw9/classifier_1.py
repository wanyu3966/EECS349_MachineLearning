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

# not use, no preprocess except flatten data matrix
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

def build_classifier(images, labels, actFn='tanh', sol='adam', batch=5, epoch=100, hiddenNodeNum=200):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. 
    classifier = MLPClassifier(activation=actFn, solver=sol, batch_size=batch, max_iter=epoch, hidden_layer_sizes=(hiddenNodeNum, 10), learning_rate_init=0.001)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set_1.p', 'w'))
    pickle.dump(training_labels, open('training_labels_1.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

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

def expForTrainSize(images, labels):
    errorList = []
    for i in range(8):
        dataNum = (i + 1) * 4000 + 20000
        crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels, numDataToUse=dataNum, numCross=10000 ,numTest=20000)
        print len(trainData), len(testData)
        classifier = build_classifier(trainData, trainLabel)
        predicted = classify(testData, classifier)
        error = error_measure(predicted, testLabel)
        errorList.append(error)
        print error
    print errorList

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

def getImage(predicted, testLabel, testData, string='classifier_1_', digit=2, imageNum=20, ):
    flag = 0
    for i in range(len(predicted)):
        if flag >= imageNum:
            break
        preLabel = predicted[i]
        trueLabel = testLabel[i]
        if trueLabel == digit and preLabel != trueLabel:
            plotFlatImage(trueLabel, testData[i], saveName=(string + str(flag) + 'predictedAs' + str(preLabel) + '.png'))
            flag += 1

def expForComp(images, labels):
    foldNum = 10
    testSetList = []
    testLabelList = []
    for i in range(foldNum):
        testSetList.append([])
        testLabelList.append([])
        crossData, crossLabel, trainData, trainLabel, testSetList[i], testLabelList[i] = dataSplit(images, labels, numDataToUse=30000, numCross=10000 ,numTest=10000)

    classifier = pickle.load(open('classifier_1.p'))
    tempErr = []
    for i in range(foldNum):
        predicted = classify(testSetList[i], classifier)
        tempErr.append(error_measure(predicted, testLabelList[i]))
    meanErr = sum(tempErr)/float(foldNum)
    print tempErr, 'mean error rate:', meanErr

    classifier = pickle.load(open('classifier_2.p'))
    tempErr = []
    for i in range(foldNum):
        predicted = classify(testSetList[i], classifier)
        tempErr.append(error_measure(predicted, testLabelList[i]))
    meanErr = sum(tempErr)/float(foldNum)
    print tempErr, 'mean error rate:', meanErr


if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(10), path='.')
    # preprocessing
    images = preprocess(images)

    # compare two classifier
    # expForComp(images, labels)

    # number of train = numDataToUse - numTest
    crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels, numDataToUse=50000, numCross=10000 ,numTest=20000)

    # to submit, set the para to best as default
    classifier = build_classifier(trainData, trainLabel)
    # save_classifier(classifier, trainData, trainLabel)
    # classifier = pickle.load(open('classifier_1.p'))
    # print 'finish build'
    predicted = classify(testData, classifier)
    # print 'finish classify'
    # getImage(predicted, testLabel, testData, digit=2, imageNum=20, string='classifier_1_')
    print error_measure(predicted, testLabel)
    print confusion_matrix(testLabel, predicted)

    # classifier = pickle.load(open('classifier_2.p'))
    # print 'finish build'
    # predicted = classify(testData, classifier)
    # print 'finish classify'
    # getImage(predicted, testLabel, testData, digit=2, imageNum=20, string='classifier_2_')
    # print error_measure(predicted, testLabel)
    # print confusion_matrix(testLabel, predicted)