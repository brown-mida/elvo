import numpy as np

from ml.preprocess_luke import standardize, bound_pixels


def test_crop():
    # TODO: The function needs to be refactored first
    pass


def test_bound_pixels():
    image = np.array([[[-1000, 500],
                       [1000, 500]],
                      [[800, 400],
                       [600, 200]]])
    expected = np.array([[[-500, 400],
                          [400, 400]],
                         [[400, 400],
                          [400, 200]]])
    actual = bound_pixels(image, -500, 400)
    assert np.array_equal(expected, actual)


def test_standardize():
    image = np.array([[[-1000, 500],
                       [1000, 500]],
                      [[800, 400],
                       [600, 200]]])
    expected = np.array([[[0., 0.75],
                          [1., 0.75]],
                         [[0.9, 0.7],
                          [0.8, 0.6]]])
    actual = standardize(image)
    assert np.array_equal(expected, actual)
