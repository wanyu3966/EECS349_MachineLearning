import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

def preprocess(class_array, n_classes):
    tempArray = np.zeros((len(class_array), n_classes))
    for i in range(len(class_array)):
        tempArray[i][class_array[i]] = 1
    return tempArray

def main():
    """
    Your main() function should read in the provided csv file
    and call your two neural networks. It should not output anything
    other than the default tflearn output.
    """
    csv_file_name = ''
    do_linear = False
    if len(sys.argv) == 3:
        csv_file_name = sys.argv[1]
        do_linear = int(sys.argv[2])

    # YOUR CODE HERE
    position_array, class_array = read_csv(csv_file_name)
    # plot_spiral(position_array, class_array, 'dataPlot')
    n_classes = len(set(class_array))
    if do_linear == 0:
        linear_classifier(position_array, class_array, n_classes)
    if do_linear == 1:
        non_linear_classifier(position_array, class_array, n_classes)


def linear_classifier(position_array, class_array, n_classes):
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
        model.fit(position_array, new_clss_array, n_epoch=1, batch_size=8, show_metric=True, snapshot_step=1)
        # print 'linear_classifier\' accuracy is: %s' %(get_accuracy(position_array, class_array, model))
        plot_spiral_and_predicted_class(position_array, class_array, model, 'linearModel', 'linear model classification')

def non_linear_classifier(position_array, class_array, n_classes):
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
        net = tflearn.fully_connected(net, 64, activation='sigmoid')
        # output node
        net = tflearn.fully_connected(net, n_classes, activation='softmax')
        
        net = tflearn.regression(net, optimizer='sgd', learning_rate=1.0, loss='categorical_crossentropy')
        # define model
        model = tflearn.DNN(net)
        # start training
        # model.fit(position_array, new_clss_array, n_epoch=10, batch_size=16, show_metric=True, snapshot_epoch=False)
        model.fit(position_array, new_clss_array, n_epoch=128, batch_size=2, show_metric=True, snapshot_step=1)
        print 'non_linear_classifier\' accuracy is: %s' %(get_accuracy(position_array, class_array, model))
        plot_spiral_and_predicted_class(position_array, class_array, model, 'nonlinearModel', 'non-linear model classification')


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
    x_min, x_max = position_array[:, 0].min() - 1, position_array[:, 0].max() + 1
    y_min, y_max = position_array[:, 1].min() - 1, position_array[:, 1].max() + 1
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
    fig.savefig(output_file_name)


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
