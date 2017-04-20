#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from gmm_est import gmm_est
import scipy.stats
import math

def resultPlot(x1, x2, result1, result2):
    plt.subplot(2, 1, 1)
    plt.xlabel('data point')
    plt.ylabel('amount')
    plt.title('class 1 of data')
    plt.ylim(0, 10)
    plt.hist(x1, len(x1), color='green')
    x1class1, x1class2 = getClassData(x1, result1)
    x1class1y = np.zeros(len(x1class1))
    x1class2y = np.zeros(len(x1class2))
    plt.plot(x1class1, x1class1y, 'ro', label='class 1')
    plt.plot(x1class2, x1class2y, 'bo', label='class 2')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('data point')
    plt.ylabel('amount')
    plt.title('class 2 of data')
    plt.ylim(0, 10)
    plt.hist(x2, len(x2), color='green')
    x2class1, x2class2 = getClassData(x2, result2)
    x2class1y = np.zeros(len(x2class1))
    x2class2y = np.zeros(len(x2class2))
    plt.plot(x2class1, x2class1y, 'ro', label='class 1')
    plt.plot(x2class2, x2class2y, 'bo', label='class 2')
    plt.legend()
    plt.show()

def getClassData(x, result):
    class1_data = []
    class2_data = []
    for i in range(len(x)):
        if result[i] == 1:
            class1_data.append(x[i])
        else:
            class2_data.append(x[i])
    return class1_data, class2_data

def getGmmErr(result1, result2):
    errCount = 0
    for classV in result1:
        if classV == 2:
            errCount += 1
    for classV in result2:
        if classV == 1:
            errCount += 1
    return errCount / float(len(result1) + len(result2))

def main():
    trainPath = 'gmm_train.csv'
    testPath = 'gmm_test.csv'
    t1, t2 = read_gmm_file(trainPath)
    x1, x2 = read_gmm_file(testPath)
    # YOUR CODE FOR PROBLEM 3 GOES HERE
    # initial value
    wt1 = [.5, .5]
    mu1 = [10.0, 30.0]
    sigmasq1 = [1.0, 1.0]
    mu_results1, sigma2_results1, w_results1, L1 = gmm_est(t1, mu1, sigmasq1, wt1, 20)
    p1 = len(t1) / float(len(t1) + len(t2))
    mu2 = [-25.0, -5.0, 50.0]
    sigmasq2 = [1.0, 1.0, 1.0]
    wt2 = [.2, .3, .5]
    mu_results2, sigma2_results2, w_results2, L2 = gmm_est(t2, mu2, sigmasq2, wt2, 20)
    # get class data
    result1 = gmm_classify(x1, mu_results1, sigma2_results1, w_results1,  mu_results2, sigma2_results2, w_results2, p1)
    result2 = gmm_classify(x2, mu_results1, sigma2_results1, w_results1,  mu_results2, sigma2_results2, w_results2, p1)
    if p1 >= 0.5:
        errorPrior = 1 - p1
    else:
        errorPrior = p1
    errorGMM = getGmmErr(result1, result2)
    print 'prior error: %s, GMM error: %s' %(errorPrior, errorGMM)
    resultPlot(x1, x2, result1, result2)    

def classfyHelper(x, normList1, normList2, wt1, wt2, p1):
    pV1 = 0.0
    pV2 = 0.0
    for k in range(len(normList1)):
        pV1 += wt1[k]*normList1[k].pdf(x)
    for k in range(len(normList2)):
        pV2 += wt2[k]*normList2[k].pdf(x)
    pV1 *= p1
    pV2 *= (1-p1)
    if pV1 >= pV2:
        return 1
    return 2

def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
    """
    Input Parameters:
        - X           : N 1-dimensional data points (a 1-by-N numpy array)
        - mu1         : means of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - sigmasq1    : variances of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - wt1         : weights of Gaussian components of the 1st class (a 1-by-K1 numpy array, sums to 1)
        - mu2         : means of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - sigmasq2    : variances of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - wt2         : weights of Gaussian components of the 2nd class (a 1-by-K2 numpy array, sums to 1)
        - p1          : the prior probability of class 1.

    Returns:
        - class_pred  : a numpy array containing results from the gmm classifier
                        (the results array should be in the same order as the input data points)
    """

    # YOUR CODE FOR PROBLEM 3 HERE
    # initial
    class_pred = []
    K1 = len(mu1)
    normList1 = []
    for k in range(K1):
        normList1.insert(k, scipy.stats.norm(mu1[k], math.sqrt(sigmasq1[k])))
    K2 = len(mu2)
    normList2 = []
    for k in range(K2):
        normList2.insert(k, scipy.stats.norm(mu2[k], math.sqrt(sigmasq2[k])))
    # compare
    for i in range(len(X)):
        class_pred.insert(i, classfyHelper(X[i], normList1, normList2, wt1, wt2, p1))
    return class_pred


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()
