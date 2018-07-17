import numpy as np
import sklearn.preprocessing

from blueno import utils


def test_create_callbacks_one_class():
    X = np.random.rand(10, 224, 224, 3)
    y = np.random.randint(0, 1, size=(10, 1))
    utils.create_callbacks(X, y, X, y, csv_file='/tmp/callbacks_test.csv')


def test_create_callbacks_three_classes():
    X = np.random.rand(10, 224, 224, 3)
    y = np.random.randint(0, 3, size=(10,))
    y = sklearn.preprocessing.label_binarize(y, [0, 1, 2])
    utils.create_callbacks(X, y, X, y, csv_file='/tmp/callbacks_test.csv')


def test_create_callbacks_no_output():
    X = np.random.rand(10, 224, 224, 3)
    y = np.random.randint(0, 3, size=(10,))
    y = sklearn.preprocessing.label_binarize(y, [0, 1, 2])
    utils.create_callbacks(X, y, X, y)


def test_create_callbacks_reduce_lr():
    X = np.random.rand(10, 224, 224, 3)
    y = np.random.randint(0, 3, size=(10,))
    y = sklearn.preprocessing.label_binarize(y, [0, 1, 2])
    utils.create_callbacks(X, y, X, y, reduce_lr=True)
