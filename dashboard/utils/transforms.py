# Code borrowed from
# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook

import numpy as np
from scipy.ndimage.interpolation import zoom


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
    intercept = 0
    for i, s in enumerate(slices):
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope
        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)
        image[i] += np.int16(intercept)

    # Some scans use -2000 as the default value for pixels not in the body
    # We set these pixels to -1000, the HU for air
    image[image == intercept - 2000] = -1000

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


def mip_normal(array, index=0):
    array = np.max(array, axis=index)
    return np.expand_dims(array, axis=2)


def mip_multichannel(array, num_slices=3, index=0):
    slices = [array.shape[0] // 3, (array.shape[0] * 2) // 3, array.shape[0]]
    array = [array[:slices[0]],
             array[slices[0]:slices[1]],
             array[slices[1]:]]
    for i in range(len(array)):
        array[i] = np.max(array[i], axis=index)
    array = np.array(array)
    return np.moveaxis(array, index, 2)


def crop_z(arr, lower, upper):
    return arr[lower:upper, :, :]


def center_crop_xy(arr, size):
    center_x = arr.shape[1] // 2
    center_y = arr.shape[2] // 2
    return arr[:, center_x - (size // 2):center_x + (size // 2),
               center_y - (size // 2):center_y + (size // 2)]


def bound_hu(arr, min_hu, max_hu):
    arr = np.clip(arr, min_hu, max_hu)
    arr[arr == min_hu] = -50
    arr[arr == max_hu] = -50
    return arr
