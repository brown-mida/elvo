import numpy as np
import scipy.ndimage


def crop(image, output_shape=(200, 200, 200)):
    """Crops the input pixel array. Centering the width and length,
    and taking the top portion in the height axis
    """
    assert image.ndim == 3
    assert all([image.shape[i] >= output_shape[i] for i in range(3)])

    for dim in range(3):
        if dim == 0:
            start_idx = 0
        else:
            start_idx = image.shape[dim] // 2 - output_shape[dim] // 2
        selected_indices = \
            list(range(start_idx, start_idx + output_shape[dim]))
        image = image.take(selected_indices, axis=dim)
    return image


def crop_z(image, z=200):
    assert image.shape[2] >= z

    selected_indices = list(range(z))
    image = image.take(selected_indices, axis=2)
    return image


def crop_center(img, cropx, cropy):
    x = img.shape[1]
    y = img.shape[2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, startx:startx + cropx, starty:starty + cropy]


def rotate_img(img):
    angle = np.random.uniform(-15, 15)
    return scipy.ndimage.rotate(img, angle)


def flip_img(img):
    return np.flipud(img)


def gaussian_img(img):
    sigma = np.random.uniform(0.2, 0.8)
    return scipy.ndimage.gaussian_filter(img, sigma)


def translated_img(img):
    x_shift = int(np.random.uniform(-20, 20))
    y_shift = int(np.random.uniform(-20, 20))
    return scipy.ndimage.shift(img, (x_shift, y_shift, 0))


def zoom_img(img):
    zoom_val = np.random.uniform(1.05, 1.20)
    return scipy.ndimage.zoom(img, (zoom_val, zoom_val, 1))
