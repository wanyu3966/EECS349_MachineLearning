import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

import json
import random

def preprocess(class_array, n_classes):
    tempArray = np.zeros((len(class_array), n_classes))
    for i in range(len(class_array)):
        tempArray[i][class_array[i]] = 1
    return tempArray

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
        dataDict[str(key)] = tempDict[key]
    return dataDict

def combineDict(epochList, batchList, actFuncList, hiddenNodeNumList):
    linearAccMap = {}
    nonLinAccMap = {}
    for e in epochList:
        linearAccMap.update(readJson('linearAccMap' + str(e) + '.json'))
        nonLinAccMap.update(readJson('nonLinAccMap' + str(e) + '.json'))
    writeToJson(linearAccMap, 'linearAccMap.json')
    writeToJson(nonLinAccMap, 'nonLinAccMap.json')

def dataSplit(position_array, class_array, foldNum):
    position_array = position_array.tolist()
    class_array = class_array.tolist()
    tempArr = [] #np.arange(len(position_array))
    for i in range(len(position_array)):
        tempArr.append(i)
    random.shuffle(tempArr)
    positionDict = {}
    classDict = {}
    foldIdx = 0
    positionDict[foldIdx] = []
    classDict[foldIdx] = []
    volume = len(position_array)/foldNum
    for index in range(len(tempArr)):
        if index != 0 and index%(volume) == 0:
            foldIdx+=1
            positionDict[foldIdx] = []
            classDict[foldIdx] = []
        positionDict[foldIdx].append(position_array[tempArr[index]])
        classDict[foldIdx].append(class_array[tempArr[index]])
    #print len(positionDict), len(classDict)
    return positionDict, classDict

def getArray(positionDict, classDict, foldNo):
    position_array = []
    class_array = []
    for i in range(len(positionDict)):
        if i == foldNo:
            continue
        # print positionDict[i], classDict[i]
        position_array.extend(positionDict[i])
        class_array.extend(classDict[i])
    testPosition = positionDict[foldNo]
    testClass = classDict[foldNo]
    return position_array, class_array, testPosition, testClass

def dataAnalysis(positionDict, classDict, n_classes, foldNum, linePath='linearAccMap.json', nonLinePath='nonLinAccMap.json', startNum=0, topNum=20):
    linearAccMap = readJson(linePath)
    nonLinAccMap = readJson(nonLinePath)
    printList = []

    i = 0
    for w in sorted(linearAccMap, key=linearAccMap.get, reverse=True):
        if i == topNum:
            break
        if i < startNum:
            i+=1
            continue
        param = w.split('.')
        crossVAcc = 0.0
        for k in range(foldNum):
            position_array, class_array, testPosition, testClass = getArray(positionDict, classDict, k)
            crossVAcc += linear_classifier(position_array, class_array, testPosition, testClass, n_classes, int(param[0]), int(param[1]))
        print i, w, linearAccMap[w], crossVAcc/float(foldNum)
        printList.append('%s, %s, %s' %(w, linearAccMap[w], crossVAcc/float(foldNum)))
        i += 1

    print '***************************************************'
    i = 0
    for w in sorted(nonLinAccMap, key=nonLinAccMap.get, reverse=True):
        if i == topNum:
            break
        if i < startNum:
            i+=1
            continue
        param = w.split('.')
        crossVAcc = 0.0
        for k in range(foldNum):
            position_array, class_array, testPosition, testClass = getArray(positionDict, classDict, k)
            crossVAcc += non_linear_classifier(position_array, class_array, testPosition, testClass, n_classes, int(param[0]), int(param[1]), param[2], int(param[3]))
        print i, w, nonLinAccMap[w], crossVAcc/float(foldNum)
        printList.append('%s, %s, %s' %(w, nonLinAccMap[w], crossVAcc/float(foldNum)))
        i += 1
    print '***************************************************'
    for printStr in printList:
        print printStr
    # lineMax = 0.0
    # lineStr = ''
    # for e in epochList:
    #     for b in batchList:
    #         tempStr = 'linear_classifier with epoch %s, batch %s, accuracy is %s' %(str(e), str(b), linearAccMap[str(e)+'.'+str(b)])
    #         print tempStr
    #         if linearAccMap[str(e)+'.'+str(b)] > lineMax:
    #             lineMax = linearAccMap[str(e)+'.'+str(b)]
    #             lineStr = tempStr
    # nonLineMax = 0.0
    # nonLineStr = ''
    # for e in epochList:
    #     for b in batchList:
    #         for a in actFuncList:
    #             for h in hiddenNodeNumList:
    #                 tempStr = 'non_linear_classifier with epoch %s, batch %s, activation %s, hiddenNodeNum %s, accuracy is %s' %(str(e), str(b), a, str(h), nonLinAccMap[str(e)+'.'+str(b)+'.'+a+'.'+str(h)])
    #                 print str(e), str(b), a, str(h), nonLinAccMap[str(e)+'.'+str(b)+'.'+a+'.'+str(h)]
    #                 if nonLinAccMap[str(e)+'.'+str(b)+'.'+a+'.'+str(h)] > nonLineMax:
    #                     nonLineMax = nonLinAccMap[str(e)+'.'+str(b)+'.'+a+'.'+str(h)]
    #                     nonLineStr = tempStr
    # print lineStr
    # print nonLineStr

def main():
    """
    Your main() function should read in the provided csv file
    and call your two neural networks. It should not output anything
    other than the default tflearn output.
    """
    csv_file_name = 'spiral_data.csv'
    position_array, class_array = read_csv(csv_file_name)
    n_classes = len(set(class_array))
    actFuncList = ['linear', 'sigmoid', 'relu', 'softmax', 'tanh']
    epochList = [1, 2, 4, 8, 16, 32, 64, 128]
    #epochList = [64, 128]
    batchList = [1, 2, 4, 8, 16, 32, 64, 128]
    hiddenNodeNumList = [1, 2, 4, 8, 16, 32, 64, 128]
    linearAccMap = {}
    nonLinAccMap = {}
    # for e in epochList:
    #     for b in batchList:
    #         linearAccMap[str(e)+'.'+str(b)] = linear_classifier(position_array, class_array, n_classes, e, b)
    #         for a in actFuncList:
    #             for h in hiddenNodeNumList:
    #                 nonLinAccMap[str(e)+'.'+str(b)+'.'+a+'.'+str(h)] = non_linear_classifier(position_array, class_array, n_classes, e, b, a, h)
    # writeToJson(linearAccMap, 'linearAccMap.json')
    # writeToJson(nonLinAccMap, 'nonLinAccMap.json')
    # combineDict(epochList, batchList, actFuncList, hiddenNodeNumList)
    positionDict, classDict = dataSplit(position_array, class_array, 10)
    # print positionDict, classDict
    dataAnalysis(positionDict, classDict, n_classes, 10, './accMap/linearAccMap.json', './accMap/nonLinAccMap.json', 0, topNum=10)



def linear_classifier(position_array, class_array, testPosition, testClass, n_classes, n_epoch=10, batch_size=16):
    """
    Here you will implement a linear neural network that will classify the input data. The input data is
    an x, y coordinate (in 'position_array') and a classification for that x, y coordinate (in 'class_array'). The
    order of the data in 'position_array' corresponds with the order of the data in 'class_array', i.e., the ith element
    in 'position_array' is classified by the ith element in 'class_array'.

    Your neural network will have an input layer that has two input nodes (an x coordinate and y coordinate)
    and an output layer that has four nodes (one for each class) with a softmax activation.

    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param n_classes: an integer that is the number of classes your data has
    """
    # linear classifier
    with tf.Graph().as_default():
        # YOUR CODE FOR PROBLEM 6A GOES HERE
        # preprocess array
        new_clss_array = preprocess(class_array, n_classes)
        # input node
        net = tflearn.input_data(shape=[None, 2])
        # output node
        net = tflearn.fully_connected(net, n_classes, activation='softmax')

        net = tflearn.regression(net, loss='categorical_crossentropy')
        # define model
        model = tflearn.DNN(net)
        # start training
        model.fit(position_array, new_clss_array, n_epoch=n_epoch, batch_size=batch_size, show_metric=True, snapshot_epoch=False)
        accuracy = get_accuracy(testPosition, testClass, model)
        print 'linear_classifier with epoch %s, batch %s, accuracy is: %s' %(n_epoch, batch_size, accuracy)
    return accuracy

def non_linear_classifier(position_array, class_array, testPosition, testClass, n_classes, n_epoch=10, batch_size=16, actFunction = 'sigmoid', hiddenNodeNum = 32):
    """
    Here you will implement a non-linear neural network that will classify the input data. The input data is
    an x, y coordinate (in 'position_array') and a classification for that x, y coordinate (in 'class_array'). The
    order of the data in 'position_array' corresponds with the order of the data in 'class_array', i.e., the ith element
    in 'position_array' is classified by the ith element in 'class_array'.

    Your neural network should have three layers total. An input layer and two fully connected layers
    (meaning that the middle layer is a hidden layer). The second fully connected layer is the output
    layer (so it should have 4 nodes and a softmax activation function). You get to decide how many
    nodes the middle layer has and the activation function that it uses.

    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param n_classes: an integer that is the number of classes your data has
    """
    with tf.Graph().as_default():
        # YOUR CODE FOR PROBLEM 6C GOES HERE:
        # preprocess class array
        new_clss_array = preprocess(class_array, n_classes)
        # input node
        net = tflearn.input_data(shape=[None, 2])
        # hidden node
        net = tflearn.fully_connected(net, hiddenNodeNum, activation=actFunction)
        # output node
        net = tflearn.fully_connected(net, n_classes, activation='softmax')

        net = tflearn.regression(net, optimizer='sgd', learning_rate=1.0, loss='categorical_crossentropy')
        # define model
        model = tflearn.DNN(net)
        # start training
        model.fit(position_array, new_clss_array, n_epoch=n_epoch, batch_size=batch_size, snapshot_step=None, snapshot_epoch=False)
        # model.fit(position_array, new_clss_array, n_epoch=10, batch_size=16, show_metric=True, snapshot_step=1, validation_set=0.1)
        accuracy = get_accuracy(testPosition, testClass, model)
        print 'non_linear_classifier with epoch %s, batch %s, activation %s, hiddenNodeNum %s, accuracy is: %s' %(n_epoch, batch_size, actFunction, hiddenNodeNum, accuracy)
    return accuracy


def plot_spiral_and_predicted_class(position_array, class_array, model, output_file_name, title):
    """
    This function plots the spirals with each position with its class colored and the space colored to show
    what the model predicts.

    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param model: a tflearn model object that will be used to color the space
    :param output_file_name: string containing a name for the output file
    :param title: title for the plot
    """
    h = 0.02
    x_min, x_max = position_array[:, 0].min() - 1, positions[:, 0].max() + 1
    y_min, y_max = position_array[:, 1].min() - 1, positions[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    z = z.reshape(xx.shape)
    plt.close('all')
    fig = plt.figure()
    plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(position_array[:, 0], position_array[:, 1], c=class_array, s=40, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title(title)
    fig.savefig(file_name)


def plot_spiral(position_array, class_array, output_file_name):
    """
    This function only plots the spirals with each position with its class colored.
    Use this to visualize the data before you run your models.

    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param output_file_name: string containing a name for the output file
    :return:
    """
    fig = plt.figure()
    plt.scatter(position_array[:, 0], position_array[:, 1], c=class_array, s=40, cmap=plt.cm.coolwarm)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    fig.savefig(output_file_name)


def get_accuracy(position_array, class_array, model):
    """
    Gets the accuracy of your model
    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param model: a tflearn model
    :return: a float in the range [0.0, 1.0]
    """
    return np.mean(class_array == np.argmax(model.predict(position_array), axis=1))


def read_csv(path_to_file):
    """
    Reads the csv file to input
    :param path_to_file: path to the csv file
    :return: a numpy array of positions, and a numpy array of classifications
    """
    position = []
    classification = []
    with open(path_to_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None) # skip the header

        for row in reader:
            position.append(np.array([float(row[0]), float(row[1])]))
            classification.append(float(row[2]))

    return np.array(position), np.array(classification, dtype='uint8')

if __name__ == '__main__':
    main()
