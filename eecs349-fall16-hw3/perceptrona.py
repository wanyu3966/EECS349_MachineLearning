import sys
import csv
import numpy as np
import scipy

def perceptrona(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.

	w = w_init
	k = 0
	converged = False
	counter = 0
	# reformat X list to [[1.0, x1], [1.0, x2], ...]
	# so that the function is yk = w0 + w1 * xk
	newX = []
	for i in range(len(X)):
		newX.insert(i, [1.0, X[i]])
	# check newX match X so that it matches Y
	for i in range(len(X)):
		if newX[i][1] != X[i]:
			print 'reformat error: at %s, X[%s] is %s, but newX[%i] is %s, not the same.' %(i, i, X[i], i, newX[i][1])
			break
	# do perceptrona
	while not converged:
		converged = True
		for i in range(len(newX)):
			if np.dot(w, newX[i]) > 0:
				result = 1
			else:
				result = -1
			if result != Y[i]:
				w = w + np.dot(newX[i], Y[i])
				converged = False
		k += 1
	return (w, k)

# get perceptrone in case we want to print the line
def f(x, w0, w1):
	return w0 + w1 * x

def main():
	# rfile = sys.argv[1]
	rfile = './linearclass.csv'
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
	w_init = [0.0, 0.0]
	w, k = perceptrona(w_init, X1, Y1)
	print w, k

if __name__ == "__main__":
	main()
