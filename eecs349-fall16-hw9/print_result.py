import pickle
import sklearn
from mnist import load_mnist
from sklearn import svm # this is an example of using SVM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier # use multi-layer perceptron algorithm (neural network)
from sklearn.model_selection import KFold

import random
import csv
import sys

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

def read_csv_file_row(file_name):
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

        data = np.array(data[0])
        data = data.astype(np.float)
        return data

def write_csv_file(file_name, data):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')

        for row in data:
            writer.writerow(row)

def plot_line(labels, result):
    fig = plt.figure()
    plt.plot(range(len(result)), result, color = 'blue')
    plt.xticks(range(len(labels)), labels, size='small')
    plt.ylabel('Mean Error Rate')
    plt.legend()
    plt.show()

def plot_lines(linear, svm):
    fig = plt.figure()
    plt.plot(range(len(linear)), linear, color = 'blue', label = 'A')
    plt.plot(range(len(svm)), svm, color='red', label='B')
    plt.xlabel('Testing Set')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    c = ['10^-5', '10^-3', '10^-1', '1', '10', '10^2', '10^3', '10^5', '10^7']

    kernel = ['linear', 'rbf', 'poly', 'sigmoid']

    gamma = ['10^-4', '10^-3', '10^-2', '10^-1', 1, '10', '10^2']

    size = ['4000', '8000', '12000', '16000', '20000', '24000', '28000', '32000']

    error_kernel = []
    error_rates = read_csv_file('./svm_kernel_exp.csv')
    for rate in error_rates:
        error_kernel.append(sum(rate) / float(len(rate)))
    plot_line(kernel, error_kernel)

    error_c = []
    error_rates = read_csv_file('./svm_c_exp.csv')
    for rate in error_rates:
        error_c.append(sum(rate)/float(len(rate)))
    plot_line(c, error_c)

    error_gamma = []
    error_rates = read_csv_file('./svm_gamma_exp.csv')
    for rate in error_rates:
        error_gamma.append(sum(rate) / float(len(rate)))
    plot_line(gamma, error_gamma)

    error_size = read_csv_file_row('./svm_size_exp.csv')
    plot_line(size, error_size)

    linear = [0.3287, 0.3272, 0.3315, 0.327, 0.3253, 0.3277, 0.3298, 0.33, 0.3274, 0.3298]
    svm = [0.1074, 0.1068, 0.1062, 0.1072, 0.1038, 0.1071, 0.1051, 0.1057, 0.1078, 0.1055]
    plot_lines(linear, svm)

