import logging
import sys
import datetime
from math import log2
from operator import itemgetter
from typing import Tuple, Callable, Iterable, Iterator
import numpy as np

Data = Tuple[np.ndarray, np.ndarray]
Measure = Callable[[Data, int, int], float]

methods = ['GINI', "CART", 'IG']

TREE_LEVELS = 2


def IG(D: Data, index, value):
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

    Dy, cy, Dn, cn = split_data((data, classes), index, value)

    # Calculate entropy of split data
    entropy_dy = get_entropy((Dy, cy))
    entropy_dy = entropy_dy if not np.isnan(entropy_dy) else 0
    entropy_dn = get_entropy((Dn, cn))
    entropy_dn = entropy_dn if not np.isnan(entropy_dn) else 0

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
    try:
        for i, val in enumerate(data[:, index]):
            if val <= value:
                yrows.append(i)
            else:
                nrows.append(i)
    except IndexError as e:
        logging.error(e)
        logging.error(data)
        logging.error(index)

    # Split captured entire dataset on one side
    if not nrows or not yrows:
        pass
        # return 0

    # Split the data
    Dy = np.take(data, yrows, axis=0)
    cy = np.take(D[1], yrows)
    Dn = np.take(data, nrows, axis=0)
    cn = np.take(D[1], nrows)

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
    Dy, cy, Dn, cn = split_data(D, index, value)
    n = X.shape[0]
    ny = cy.size
    nn = cn.size

    gy = _GI(cy)
    gn = _GI(cn)
    g_total = ny / n * gy + nn / n * gn

    return g_total


def _GI(classes):
    prob = class_probability(classes)
    return 1 - ((prob ** 2) + ((1 - prob) ** 2))


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

    summation = abs(p_cy - p_cn) + abs((1 - p_cy) - (1 - p_cn))
    cart = 2 * (ny / n) * (nn / n) * summation
    return cart


def bestSplit(D: Data, criterion: str):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """
    # Get our measure function and aggregate (either max() or min()) from the dispatcher
    measure, aggregate = dispatcher(criterion)

    # Call the aggregate function on our iterable telling it to compare the value of each tuple returned
    iterable = split_generator(D, measure)
    index, score, value = aggregate(iterable, key=itemgetter(1))
    return index, value


def split_generator(D: Data, measure: Measure) -> Iterator[Tuple[int, float, int]]:
    """
    Use a generator to return an iterator over all values in the matrix. Return a tuple with the row index
    plus the value of the current cell.
    :param D: Data tuple
    :param measure: Callable referring to the measure calculating the effectiveness of our split
    :yi
    """
    dims = D[0].shape
    data = D[0]
    full_data = [(col, measure(D, col, data[row, col]), data[row, col]) for row in range(dims[0]) for col in
                 range(dims[1])]
    return full_data
    # Generators are real slow lol
    # for row in range(dims[0]):
    #     for col in range(dims[1]):
    #         logging.debug(f'row {row} col{col} value{data[row,col]}')
    #         full_data.append(col, measure(D, col, data[row,col] ))
    #         # yield col, measure(D, col, data[row, col]), data[row,col]


def load(filename: str) -> Tuple[np.ndarray, np.ndarray]:
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
    criterion = 'IG'
    dTree = DecisionTree(criterion)
    dTree.gen_tree(train, TREE_LEVELS)

    classification = [int(dTree.classify(row)) for row in test[0]]

    logging.info(f"Predicted classes for test data: {classification}")
    logging.info(f"Actual classification:           {test[1]} ")
    return classification


def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    criterion = 'GINI'
    dTree = DecisionTree(criterion)
    dTree.gen_tree(train, TREE_LEVELS)

    classification = [int(dTree.classify(row)) for row in test[0]]

    logging.info(f"Predicted classes for test data: {classification}")
    logging.info(f"Actual classification:           {test[1]} ")
    return classification


def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    criterion = 'CART'
    dTree = DecisionTree(criterion)
    dTree.gen_tree(train,TREE_LEVELS)

    classification = [int(dTree.classify(row)) for row in test[0]]

    logging.info(f"Predicted classes for test data: {classification}")
    logging.info(f"Actual classification::          {test[1]} ")
    return classification


def dispatcher(name: str) -> Tuple[Measure,
                                   Callable[[Iterable, Callable], float]]:
    """
    Return the callable associated with the given name. Also returns the correct comparator function
    associated with each measure.
    :param name: name of the function we are looking to execute
    :return: evaluation measure, comparator
    """
    registry = {
        "IG": (IG, max),
        "GINI": (G, min),
        "CART": (CART, max)
    }
    return registry[name]


class Node:
    def __init__(self):
        self._left: Node = None
        self._right: Node = None
        self._split_value = None
        self._split_index = None
        self._class_prob = None

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, n: 'Node'):
        self._left = n

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value: 'Node'):
        self._right = value

    @property
    def value(self):
        return self._split_value

    @value.setter
    def value(self, val):
        self._split_value = val

    @property
    def index(self):
        return self._split_index

    @index.setter
    def index(self, val):
        self._split_index = val

    @property
    def prob(self):
        return self._class_prob

    @prob.setter
    def prob(self, val):
        self._class_prob = val


class DecisionTree:
    def __init__(self, criterion: str):
        self._root: Node = Node()
        self._eval = bestSplit
        self._criterion = criterion

    def gen_tree(self, D: Data, levels):
        logging.info(f'Training decision tree using {self._criterion} evaluator')
        self.decision_tree(D, self._root, levels)

    def eval(self, data: Data) -> Tuple[int, int]:
        return self._eval(data, self._criterion)

    def decision_tree(self, D: Data, curr: Node, levels):
        prob = class_probability(D[1])
        if not D or D[1].size <= 1 or prob == 1 or prob == 0 or levels == 0:  # probably need to do something else here
            logging.debug('break')
            return
        try:
            index, value = self.eval(D)
            Dy, cy, Dn, cn = split_data(D, index=index, value=value)
            # Check for split that didn't modify original data
            if np.array_equal(Dy, D[0]) or np.array_equal(Dn, D[0]):
                return
            curr.index = index
            curr.value = value
            curr.prob = prob

            curr.left = Node()
            curr.right = Node()
            logging.debug(f'Left: {Dy.shape[0]}')
            self.decision_tree((Dy, cy), curr.left, levels-1)
            logging.debug(f'Right: {Dn.shape[0]}')
            self.decision_tree((Dn, cn), curr.right, levels-1)
        except AttributeError as e:
            logging.error(e)
            logging.error(D)
            logging.error(self)

    def classify(self, row: np.ndarray):
        return self._rClassify(self._root, row)

    def _rClassify(self, curr: Node, row: np.ndarray):

        try:
            val = row[curr.index]
            c = curr.prob >= 0.500000
        except (TypeError, AttributeError):
            return None
        if val <= curr.value:
            left = self._rClassify(curr.left, row)
            return left if left is not None else c
        else:
            right = self._rClassify(curr.right, row)
            return right if right is not None else c


def main():
    """This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point
    unlike C, Java, etc.
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
    """
    contents = load('train.txt')
    test = load('test.txt')
    train = contents
    results = []

    logging.info('running classify')
    results.append(classifyIG(train, test))
    logging.info('done running classify')

    logging.info('\n\nBeginning CART classify')
    results.append(classifyCART(train, test))
    logging.info('Finished CART classify')

    logging.info('\n\nBeginning Ginni classify')
    results.append(classifyG(train, test))
    logging.info('Finished Ginni classify')


    logging.info('\n')
    logging.info('Finished all three classifiers')
    logging.info('Results:')
    logging.info(f'Information Gain: {results[0]}')
    logging.info(f'CART:             {results[1]}')
    logging.info(f'Gini Index:       {results[2]}')
    logging.info(f'Actual:           {test[1]}')
    return
    i, v = 2, 0
    split_method = "GINI"
    gain = IG(contents, i, v)
    ginni = G(contents, i, v)

    index, value = bestSplit(contents, split_method)
    logging.info(f'Dataset:\n{contents[0]}')
    logging.info(f'Gain from a split on column {i} at value {v} is: +{gain:.4f}')
    logging.info(f'Ginni index split on column {i} at value {v} is {ginni:.4f}')

    optimized = []

    # start = datetime.datetime.now()
    # for method in methods:
    #     index,value =bestSplit(contents, method)
    #     optimized.append((method, index, value))
    # end = datetime.datetime.now()
    #
    # diff = end-start
    # logging.info(f'Calculated best split for GINI IG & CART in {diff}')
    # for method, index, value in optimized:
    #     logging.info(f'\n\nCalculated best {method} split on input data with shape {contents[0].shape}\n'
    #                  f'Best split is on attribute index: {index} with value {value}')


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

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    main()
