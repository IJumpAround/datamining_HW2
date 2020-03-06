import numpy as np
from math import log2


def IG(D: (np.ndarray,np.ndarray), index, value):
	"""Compute the Information Gain of a split on attribute index at value
	for dataset D.
	
	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Information Gain for the given split
	"""
	entropy_0 = get_entropy(D)
	n = D[0].shape[0]
	yrows = []
	nrows = []
	data = D[0]

	# Find row indices to separate
	for i, val in enumerate(data[:,index]):
		if val <= value:
			yrows.append(i)
		else:
			nrows.append(i)

	# Split captured entire dataset on one side
	if not nrows or not yrows:
		return 0

	# Split the data
	Dy = np.take(data, yrows, axis=0)
	cy = np.take(D[1], yrows)
	Dn = np.take(data, nrows, axis=0)
	cn = np.take(D[1], nrows)

	# Calculate entropy of split data
	entropy_dy = get_entropy((Dy,cy))
	entropy_dn = get_entropy((Dn,cn))

	ny= cy.size
	nn = cn.size

	# Calculate gain
	gain = entropy_0 - ny/n*entropy_dy + nn/n*entropy_dn
	return gain

def get_entropy(D: (np.ndarray, np.ndarray)):
	"""
	Determine the entropy of the dataset
	:param D:
	"""
	prob = class_probability(D[1])
	entropy = -(prob*log2(prob) + (1-prob)*log2(1-prob))
	return entropy


def class_probability(c: np.ndarray):
	"""
	Since the classifier is binary, we can sum the 1s and 0s then divide by the count to find the probability
	:param c:
	:return:
	"""
	return c.sum() / c.size


def G(D, index, value):
	"""Compute the Gini index of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Gini index for the given split
	"""


def CART(D, index, value):
	"""Compute the CART measure of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the CART measure for the given split
	"""


def bestSplit(D, criterion):
	"""Computes the best split for dataset D using the specified criterion

	Args:
		D: A dataset, tuple (X, y) where X is the data, y the classes
		criterion: one of "IG", "GINI", "CART"

	Returns:
		A tuple (i, value) where i is the index of the attribute to split at value
	"""


# functions are first class objects in python, so let's refer to our desired criterion by a single name


def load(filename) -> (np.ndarray, np.ndarray):
	"""Loads filename as a dataset. Assumes the last column is classes, and
	observations are organized as rows.

	Args:
		filename: file to read

	Returns:
		A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
		where X[i] comes from the i-th row in filename; y is a list or ndarray of 
		the classes of the observations, in the same order
	"""

	matrix = np.genfromtxt(filename, delimiter=',')
	classes = np.copy(matrix[:, -1])
	matrix = np.delete(matrix, -1, axis=1)

	return (matrix, classes)


def classifyIG(train, test):
	"""Builds a single-split decision tree using the Information Gain criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""


def classifyG(train, test):
	"""Builds a single-split decision tree using the GINI criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""


def classifyCART(train, test):
	"""Builds a single-split decision tree using the CART criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""


def main():
	"""This portion of the program will run when run only when main() is called.
	This is good practice in python, which doesn't have a general entry point 
	unlike C, Java, etc. 
	This way, when you <import HW2>, no code is run - only the functions you
	explicitly call.
	"""


if __name__ == "__main__":
	"""__name__=="__main__" when the python script is run directly, not when it 
	is imported. When this program is run from the command line (or an IDE), the 
	following will happen; if you <import HW2>, nothing happens unless you call
	a function.
	"""
	# main()
	contents = load('train.txt')
	i, v = 1, 35
	gain = IG(contents, i, v)

	print(f'Gain from a split on column {i} at value {v} is: +{gain}')
