import typing

import numpy as np
import scipy.ndimage


def rotate_img(img: np.ndarray) -> np.ndarray:
    angle = np.random.uniform(-15, 15)
    return scipy.ndimage.rotate(img, angle)


def flip_img(img: np.ndarray) -> np.ndarray:
    return np.flipud(img)


def gaussian_img(img: np.ndarray) -> np.ndarray:
    sigma = np.random.uniform(0.2, 0.8)
    return scipy.ndimage.gaussian_filter(img, sigma)


def translated_img(img: np.ndarray) -> np.ndarray:
    x_shift = int(np.random.uniform(-20, 20))
    y_shift = int(np.random.uniform(-20, 20))
    return scipy.ndimage.shift(img, (x_shift, y_shift, 0))


def zoom_img(img: np.ndarray) -> np.ndarray:
    zoom_val = np.random.uniform(1.05, 1.20)
    return scipy.ndimage.zoom(img, (zoom_val, zoom_val, 1))


def bound_pixels(arr: np.ndarray,
                 min_bound: float,
                 max_bound: float) -> np.ndarray:
    arr[arr < min_bound] = min_bound
    arr[arr > max_bound] = max_bound
    return arr


def filter_pixels(arr: np.ndarray,
                  min_bound: float,
                  max_bound: float,
                  filter_value: float) -> np.ndarray:
    arr[arr < min_bound] = filter_value
    arr[arr > max_bound] = filter_value
    return arr


def average_intensity_projection():
    raise NotImplementedError()


def distance_intensity_projection():
    raise NotImplementedError()


def crop(image3d: np.ndarray,
         output_shape: typing.Tuple[int, int, int],
         height_offset=30) -> np.ndarray:
    """
    Crops a 3d image in ijk form (height as axis 0).

    :param image3d:
    :param output_shape:
    :param height_offset:
    :return:
    """
    assert image3d.ndim == 3
    assert image3d.shape[1] == image3d.shape[2]
    assert output_shape[1] == output_shape[2]
    assert output_shape[1] <= image3d.shape[1]

    lw_center = image3d.shape[1] // 2
    lw_min = lw_center - output_shape[1] // 2
    lw_max = lw_center + output_shape[1] // 2
    for i in range(len(image3d) - 1, 0, -1):
        if image3d[i, lw_center, lw_center] >= 0:
            height_max = i - height_offset
            break
    else:
        raise ValueError('Failed to a relevant pixel'
                         ' with CT value of at least zero')
    height_min = height_max - output_shape[0]

    cropped = image3d[height_min:height_max, lw_min:lw_max, lw_min:lw_max]
    assert cropped.shape == output_shape
    return cropped
