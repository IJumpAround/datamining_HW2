import logging
import sys
from math import log2
from typing import Tuple, Callable

import numpy as np





def IG(D: (np.ndarray, np.ndarray), index, value):
    """Compute the Information Gain of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Information Gain for the given split
    """
    # Get initial entropy of the data
    entropy_0 = get_entropy(D)
    data = D[0]
    n = data.shape[0]
    classes = D[1]

    Dy,cy,Dn,cn = split_data((data,classes), index, value)

    # Calculate entropy of split data
    entropy_dy = get_entropy((Dy, cy))
    entropy_dn = get_entropy((Dn, cn))
    logging.debug(f'entropy_0:  {entropy_0}')
    logging.debug(f'entropy dn: {entropy_dn}')
    logging.debug(f'entropy dy: {entropy_dy}')

    # Count of each partition
    ny = cy.size
    nn = cn.size

    # Calculate gain
    gain = entropy_0 - (ny / n * entropy_dy + nn / n * entropy_dn)
    return gain


def get_entropy(D: (np.ndarray, np.ndarray)):
    """
    Determine the entropy of the dataset
    :param D:
    """
    prob = class_probability(D[1])

    try:
        entropy = -(prob * log2(prob) + (1 - prob) * log2(1 - prob))
    except ValueError as e:
        entropy = 0
    return entropy


def split_data(D: Tuple[np.ndarray, np.ndarray], index, value) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a data set with binary classifiers split the data set such we have two data sets after splitting.
    The first contains all rows of D where the value of attribute[index]  <= value
    The second contains all rows where the specified attribute > value
    :param D: 2 tuple (data,classifier)
    :param index: Column attribute to test against value
    :param value: target value to split against
    :return: 4 tuple of the form (Dy,cy,Dn,cn)
    """

    data = D[0]
    yrows = []
    nrows = []
    for i, val in enumerate(data[:, index]):
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

    logging.debug(f'Original data:\n'
                  f'{D[0]}\n'
                  f'{D[1]}\n')
    logging.debug(f'Resulting split:\n'
                  f'Dy:\n {Dy}\n'
                  f'Dn:\n {Dn}\n'
                  f'cy:\n {cy}\n'
                  f'cn:\n {cn}\n')

    return Dy, cy, Dn, cn


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
    X = D[0]  # data
    y = D[1]  # classes

    # Get split and relevant info
    Dy, cy, Dn, cn = split_data((X,y), index, value)
    n = X.shape[0]
    ny = cy.size
    nn = cn.size

    gy = _GI(cy)
    gn = _GI(cn)
    g_total = ny/n*gy + nn/n*gn

    logging.debug(f'Calculated Gini index of this split is: {g_total}')
    return g_total

def _GI(classes):
    prob = class_probability(classes)
    return 1-(prob**2 + ((1-prob)**2))


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
    classes = D[1]
    n = classes.size

    Dy, cy, Dn, cn = split_data(D, index, value)

    p_cy = class_probability(cy)
    p_cn = class_probability(cn)
    ny = cy.size
    nn = cn.size

    summation = abs(p_cy - p_cn) + abs((1-p_cy) - (1-p_cn))
    cart = 2 * (ny/n)* (nn/n) * summation
    return cart


def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """



# functions are first class objects in python, so let's refer to our desired criterion by a single name


def load(filename)-> Tuple[np.ndarray, np.ndarray]:
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
    contents = load('test.txt')
    i, v = 1, 35
    gain = IG(contents, i, v)
    ginni = G(contents, i, v)
    logging.info(f'Gain from a split on column {i} at value {v} is: +{gain:.4f}')
    logging.info(f'Ginni index split on column {i} at value {v} is {ginni:.4f}')


def dispatcher(name: str) -> Callable:
    """
    Return the callable associated with the given name
    :param name: name of the function we are looking to execute
    :return: The correct function
    """
    registry = {
        "IG": IG,
        "GINI": G,
        "CART": CART
    }
    return registry[name]


if __name__ == "__main__":
    """__name__=="__main__" when the python script is run directly, not when it 
    is imported. When this program is run from the command line (or an IDE), the 
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
    """

    try:
        debug = sys.argv[1]
    except IndexError:
        debug = False
    print(f'Debug mode: {debug}')

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    main()


