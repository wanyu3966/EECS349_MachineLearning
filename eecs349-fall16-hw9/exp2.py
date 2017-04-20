import pickle
import sklearn
from mnist import load_mnist
from sklearn import svm # this is an example of using SVM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import random
import csv
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

def build_classifier(images, labels, c, kernel, gamma):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.

    # for MLPClassifier
    # c : [1e-5, 1e-3, 1e-1, 1.0, 1e3]
    # kernel: ['linear', 'rbf', 'poly', 'sigmoid']
    # degree: [1, 2, 8, 32, 128, 512]
    classifier = svm.SVC(C = c, kernel = kernel, gamma=gamma)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set.p', 'w'))
    pickle.dump(training_labels, open('training_labels.p', 'w'))


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

def read_csv_file(file_name):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param file_name: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    with open(file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        data = []

        for row in reader:
            data.append(row)

        data = np.array(data)
        data = data.astype(np.float)
        return data

def write_csv_file(file_name, data):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')

        for row in data:
            writer.writerow(row)

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
    c = [1e-5, 1e-3, 1e-1, 1.0, 1e3, 1e5, 1e7]
    c = np.array(c)
    c = c.astype(np.float)

    kernel = ['linear', 'rbf', 'poly', 'sigmoid']

    gamma = [1e-4, 1e-3, 1e-2, 'auto', 1, 1e1, 1e2]

    '''
    # C experiment
    result_list = []
    for k in kernel:
        print 'kernel:',k
        temp_error = []
        for i in range(foldNum):
            classifier = build_classifier(trainSetList[i], trainLabelList[i], 1.0, k, 'auto')
            predicted = classify(testSetList[i], classifier)
            temp_error.append(error_measure(predicted, testLabelList[i]))
        result_list.append(temp_error)

    write_csv_file('./svm_kernel_exp.csv', result_list)
    '''

    '''
    # kernel experiment
    result_list = []
    for C in c:
        print 'C:', C
        temp_error = []
        for i in range(foldNum):
            classifier = build_classifier(trainSetList[i], trainLabelList[i], C, 'rbf', 'auto')
            predicted = classify(testSetList[i], classifier)
            temp_error.append(error_measure(predicted, testLabelList[i]))
        result_list.append(temp_error)

    write_csv_file('./svm_c_exp.csv', result_list)
    '''

    # gamma experiment
    result_list = []
    for g in gamma:
        print 'gamma:', g
        temp_error = []
        for i in range(foldNum):
            classifier = build_classifier(trainSetList[i], trainLabelList[i], 1, 'rbf', 'auto')
            predicted = classify(testSetList[i], classifier)
            temp_error.append(error_measure(predicted, testLabelList[i]))
        result_list.append(temp_error)

    write_csv_file('./svm_c_exp.csv', result_list)

    return result_list

def svm_CK_grid_search(data, label):
    classifier = svm.SVC()

    # Set the parameters by cross-validation
    tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 'auto', 1, 1e1, 1e2], 'C': [1, 10, 100]}

    #tuned_parameters = {'kernel': ['rbf'], 'gamma': ['auto'], 'C': [1]}

    print("# Tuning hyper-parameters")
    print

    clf = GridSearchCV(classifier, tuned_parameters, cv=10)
    clf.fit(data, label)

    print("Best parameters set found on development set:")
    print
    print(clf.best_params_)
    print
    print("Grid scores on development set:")
    print
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print

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
    #images, labels = load_mnist(digits=range(10), path='.')
    # preprocessing
    #images = preprocess(images)

    #write_csv_file('./images.csv', images)

    #images = read_csv_file('./images.csv')

    # Code for loading data
    images, labels = load_mnist(digits=range(10), path='.')
    # preprocessing
    images = preprocess(images)

    crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels, numDataToUse=50000, numCross=10000, numTest=20000)
    classifier = pickle.load(open('classifier_2.p'))
    print 'finish build'
    predicted = classify(testData, classifier)
    print 'finish classify'
    flag = 0
    for i in range(len(predicted)):
        if flag > 20:
            break
        preLabel = predicted[i]
        trueLabel = testLabel[i]
        if trueLabel == 2 and preLabel != 2:
            plotFlatImage(trueLabel, testData[i],
                          saveName=('c2MissData' + str(flag) + 'predictedAs' + str(preLabel) + '.png'))
            flag += 1
    print error_measure(predicted, testLabel)
    print confusion_matrix(testLabel, predicted)

    '''
    sizes = [4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]
    # training set size experiment
    result_list = []
    for size in sizes:
        print 'size:', size
        crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels, numCross=size)
        trainSetList, trainLabelList, testSetList, testLabelList = getCrossSet(crossData, crossLabel)
        classifier = build_classifier(crossData, crossLabel, 1, 'rbf', 'auto')
        predicted = classify(testData, classifier)
        result_list.append(error_measure(predicted, testLabel))

    write_csv_file('./svm_size_exp.csv', result_list)
    '''
    '''
    crossData, crossLabel, trainData, trainLabel, testData, testLabel = dataSplit(images, labels)
    trainSetList, trainLabelList, testSetList, testLabelList = getCrossSet(crossData, crossLabel)

    result = expForPara(trainSetList, trainLabelList, testSetList, testLabelList)

    print result

    #svm_CK_grid_search(crossData, crossLabel)
    '''




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
