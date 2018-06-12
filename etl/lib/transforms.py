# Code borrowed from
# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate, zoom, shift


def get_pixels_hu(slices):
    """Takes in a list of dicom datasets and returns the 3D pixel
    array in Hounsfield scale, taking slope and intercept into
    account.
    """
    for s in slices:
        assert s.Rows == 512
        assert s.Columns == 512

    image = np.stack([np.frombuffer(s.pixel_array, np.int16).reshape(512, 512)
                      for s in slices])

    # Convert the pixels Hounsfield units (HU)
    for i, s in enumerate(slices):
        intercept = s.RescaleIntercept
        assert intercept == -1024
        slope = s.RescaleSlope
        assert slope == 1
        image[i] += np.int16(intercept)

    # Some scans use -2000 as the default value for pixels not in the body
    # We set these pixels to -1000, the HU for air
    image[image == -1024 - 2000] = -1000

    return image


def standardize_spacing(image, slices):
    """Takes in a 3D image and interpolates the image so
    each pixel corresponds to approximately a 1x1x1 box.
    """
    # Determine current pixel spacing
    spacing = np.array(
        [slices[0].SliceThickness] + list(slices[0].PixelSpacing),
        dtype=np.float32
    )
    new_shape = np.round(image.shape * spacing)
    resize_factor = new_shape / image.shape

    return zoom(image, resize_factor, mode='nearest')


def normalize(image, lower_bound=None, upper_bound=None):
    # TODO: This is an issue, we can't zero center per image
    if lower_bound is None:
        lower_bound = image.min()
    if upper_bound is None:
        upper_bound = image.max()

    image[image > upper_bound] = upper_bound
    image[image < lower_bound] = lower_bound

    return (image - image.mean()) / image.std()


def crop(image, output_shape=(200, 200, 200)):
    """Crops the input pixel array. Centering the width and length, and taking
    the top portion in the height axis"""
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
    return rotate(img, angle)


def flip_img(img):
    return np.flipud(img)


def gaussian_img(img):
    sigma = np.random.uniform(0.2, 0.8)
    return gaussian_filter(img, sigma)


def translated_img(img, dims=3):
    x_shift = int(np.random.uniform(-20, 20))
    y_shift = int(np.random.uniform(-20, 20))
    if dims == 3:
        return shift(img, (x_shift, y_shift, 0))
    else:
        return shift(img, (x_shift, y_shift))


def zoom_img(img, dims=3):
    zoom_val = np.random.uniform(1.05, 1.20)
    if dims == 3:
        return zoom(img, (zoom_val, zoom_val, 1))
    else:
        return zoom(img, (zoom_val, zoom_val))
