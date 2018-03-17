# Code borrowed from
# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook

import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.stats


def get_pixels_hu(slices):
    """Takes in a list of dicom datasets and returns the 3D pixel
    array in Houndsworth units, taking slope and intercept into
    account.
    """
    for s in slices:
        assert s.Rows == 512
        assert s.Columns == 512

    image = np.stack([np.frombuffer(s.pixel_array, np.int16).reshape(512, 512)
                      for s in slices])

    # Convert the pixels Hounsfield units (HU)
    for i, s in enumerate(slices):
        assert s.RescaleType == 'HU'
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
    return scipy.ndimage.interpolation.zoom(image,
                                            resize_factor,
                                            mode='nearest')


def normalize(image, lower_bound=None, upper_bound=None):
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
    assert all([image.shape[i] > output_shape[i] for i in range(3)])

    for dim in range(3):
        if dim == 0:
            start_idx = output_shape[dim] - output_shape[dim]
        else:
            start_idx = image.shape[dim] // 2 - output_shape[dim] // 2
        selected_indices = list(range(start_idx, start_idx + output_shape[dim]))
        image = image.take(selected_indices, axis=dim)
    return image
