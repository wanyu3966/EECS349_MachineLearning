#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats
import math
import copy

def dataPlot(x1, x2):
    plt.subplot(2, 1, 1)
    plt.hist(x1, len(x1))
    plt.title('class 1')
    plt.xlabel('data point')
    plt.ylabel('amount')
    plt.subplot(2, 1, 2)
    plt.hist(x2, len(x2))
    plt.title('class 2')
    plt.xlabel('data point')
    plt.ylabel('amount')
    plt.show()
    
def likelihoodPlot(L1, L2):
    plt.title('likelihood')
    plt.xlabel('round number')
    plt.ylabel('log likelihood value')
    plt.plot(L1, label='class 1')
    plt.plot(L2, label='class 2')
    plt.legend()
    plt.savefig('problem2:loglikelihood.png')

def main():
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    file_path = sys.argv[1]
    x1, x2 = read_gmm_file(file_path)
    
    # YOUR CODE FOR PROBLEM 2 GOES HERE
    #dataPlot(x1, x2)
    #initial parameter
    # k = 2
    # mu1 = np.full(k, sum(x1) / len(x1))
    # sigmasq1 = np.full(k, sum(np.power(x1 - mu1[0], 2))/ len(x1))
    # wt1 = np.full(k, 1.0/k)
    wt1 = [.5, .5]
    mu1 = [10.0, 30.0]
    sigmasq1 = [1.0, 1.0]
    #print mu1, sigmasq1
    mu_results1, sigma2_results1, w_results1, L1 = gmm_est(x1, mu1, sigmasq1, wt1, 20)

    # k = 3
    # mu2 = np.full(k, sum(x2) / len(x2))
    # sigmasq2 = np.full(k, sum(np.power(x2 - mu2[0], 2))/ len(x2))
    # wt2 = np.full(k, 1.0/k)
    mu2 = [-25.0, -5.0, 50.0]
    sigmasq2 = [1.0, 1.0, 1.0]
    wt2 = [.2, .3, .5]
    mu_results2, sigma2_results2, w_results2, L2 = gmm_est(x2, mu2, sigmasq2, wt2, 20)

    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu_results1, '\nsigma^2 =', sigma2_results1, '\nw =', w_results1

    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print '\nClass 2'
    print 'mu =', mu_results2, '\nsigma^2 =', sigma2_results2, '\nw =', w_results2

    # likelihood plot
    likelihoodPlot(L1, L2)

def gmm_est(X, mu_init, sigmasq_init, wt_init, its):
    """
    Input Parameters:
      - X             : N 1-dimensional data points (a 1-by-N numpy array)
      - mu_init       : initial means of K Gaussian components (a 1-by-K numpy array)
      - sigmasq_init  : initial  variances of K Gaussian components (a 1-by-K numpy array)
      - wt_init       : initial weights of k Gaussian components (a 1-by-K numpy array that sums to 1)
      - its           : number of iterations for the EM algorithm

    Returns:
      - mu            : means of Gaussian components (a 1-by-K numpy array)
      - sigmasq       : variances of Gaussian components (a 1-by-K numpy array)
      - wt            : weights of Gaussian components (a 1-by-K numpy array, sums to 1)
      - L             : log likelihood
    """

    # YOUR CODE FOR PROBLEM 1 HERE
    K = len(mu_init)
    N = len(X)
    L = np.zeros(its + 1)
    mu = np.array(copy.deepcopy(mu_init))
    sigmasq = np.array(copy.deepcopy(sigmasq_init))
    # sigma = np.sqrt(sigmasq)
    wt = np.array(copy.deepcopy(wt_init))

    for index in range(its):
        # initial normList
        if index == 0:
            normList = []
            for k in range(K):
                normList.insert(k, scipy.stats.norm(mu[k], math.sqrt(sigmasq[k])))
            # calculate initial likelihood
            for i in range(N):
                innerSum = 0.0
                for j in range(K):
                #innerSum[j] = wt[j] * normList[j].pdf(X[i])
                    innerSum += wt[j] * normList[j].pdf(X[i])
                L[0] += np.log(innerSum)
            #print 'initial likelihood is:', L[0]
        gamma = np.zeros((K, N))
        bigGamma = np.zeros(K)
        for j in range(K):
            for n in range(N):
                denominator = 0.0
                for k in range(K):
                    # use probability dense function
                    denominator += wt[k]*normList[k].pdf(X[n])
                if denominator == 0:
                    print 'error: denominator = 0 at gamma[%s][%s]' %(j, n)
                    break
                #print denominator
                gamma[j][n] = (wt[j]*normList[j].pdf(X[n])) / denominator
                #print gamma[j][n]
            #for n in range(N):
                bigGamma[j] += gamma[j][n]
        #print gamma
        #print bigGamma
        # updates new wt, mu, and sigasq
        for j in range(K):
            tempMu = 0.0
            for i in range(N):
                tempMu += gamma[j][i] * X[i]
            tempMu /= bigGamma[j]
            # not sure use new mu (tempMu) to cal sigmasq or old mu (mu[j])
            tempSigmasq = 0.0
            for i in range(N):
                tempSigmasq += gamma[j][i] * math.pow(X[i] - tempMu, 2)
            tempSigmasq /= bigGamma[j]
            if tempSigmasq < 0.05:
                print 'sigmasq[%s] is %s, to avoid overfitting, stop updating model' %(j, tempSigmasq)
                break

            wt[j] = bigGamma[j] / N
            mu[j] = tempMu
            sigmasq[j] = tempSigmasq
        if np.sum(wt) != 1:
            #print 'warning: sum of weight is %s, may not equal to 1' %(sum(wt))
            pass
                   
        # cal log likelihood
        # print K, mu, sigmasq
        # update normlist
        # normList = []
        for k in range(K):
            #normList.insert(k, scipy.stats.norm(mu[k], math.sqrt(sigmasq[k])))
            normList[k] = scipy.stats.norm(mu[k], math.sqrt(sigmasq[k]))
        for i in range(N):
            #innerSum = np.zeros(K)
            innerSum = 0.0
            for j in range(K):
                #innerSum[j] = wt[j] * normList[j].pdf(X[i])
                innerSum += wt[j] * normList[j].pdf(X[i])
            L[index + 1] += np.log(innerSum)
        #print 'round:', index, 'mu:', mu.tolist(), 'sigma sq:', sigmasq.tolist(), 'weight:', wt.tolist()
    #print 'likelihood:', L
    return mu, sigmasq, wt, L


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
