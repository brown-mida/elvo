"""
Code for preprocessing 3D CT data -- conversions to HU, interpolation, etc.
From: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook
"""
import numpy as np
from scipy.ndimage.interpolation import zoom


def get_pixels_hu(slices):
    """
    Takes in a list of dicom datasets and returns the 3D pixel array in
    Hounsfield scale, taking slope and intercept into account.

    :param slices: 3D DICOM image to process
    :return:
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
    """
    Takes in a 3D image and interpolates the image so each pixel corresponds to
    approximately a 1x1x1 box.

    :param image: Non-interpolated array
    :param slices: actual DICOM slices with spacing info
    :return:
    """
    # Determine current pixel spacing
    spacing = np.array(
        [slices[0].SliceThickness] + list(slices[0].PixelSpacing),
        dtype=np.float32
    )
    new_shape = np.round(image.shape * spacing)
    resize_factor = new_shape / image.shape

    # zoom to the right size
    return zoom(image, resize_factor, mode='nearest')


def mip_normal(array: np.ndarray) -> np.ndarray:
    """
    MIP image in the simplest, most intuitive way

    :param array: array to MIP
    :return: MIPed array
    """
    return np.max(array, axis=0)


def mip_multichannel(array: np.ndarray) -> np.ndarray:
    """
    Perform a multichannel MIP -- 3 channels to simulate normal image data

    :param array: array to MIP
    :return: MIPed array
    """
    # Get array of zeros of the desired size
    num_slices = 3
    to_return = np.zeros((num_slices, len(array[0][0]), len(array[0][0][0])))

    # channel-wise max on each slice of the array and put into the return var
    for i in range(num_slices):
        to_return[i] = np.max(array[i], axis=0)
    return to_return


def mip_overlap(array: np.ndarray) -> np.ndarray:
    """
    Perform a multichannel, overlapping MIP -- 20 overlapping channels to
    simulate how radiologists actually read scans.

    :param array: array to MIP
    :return: MIPed array
    """
    # Get array of zeros of the desired size
    num_slices = 20
    to_return = np.zeros((num_slices, len(array[0][0]), len(array[0][0][0])))

    # channel-wise max on each slice of the array and put into the return var
    for i in range(num_slices):
        to_return[i] = np.max(array[i], axis=0)
    return to_return


def crop_normal_axial(arr: np.ndarray, whence: str):
    """
    Crops the input array into the desired shape to be MIPed

    :param arr: array to be cropped
    :param whence: where it's coming from
    :return: cropped array
    """
    # from numpy
    if whence == 'numpy/axial':
        to_return = arr[len(arr) - 35 - 64:len(arr) - 35]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


def crop_normal_axial_fa(arr: np.ndarray, whence: str):
    """
    Crops the input array normally, but lower down for failure analysis

    :param arr: array to be cropped
    :param whence: where it's coming from
    :return: cropped array
    """
    # from numpy
    if whence == 'numpy/axial':
        to_return = arr[len(arr) - 50 - 64:len(arr) - 50]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


def crop_normal_coronal(arr: np.ndarray, whence: str):
    """
    Crops coronal scans to be single-channel MIPed

    :param arr: coronal array to be cropped
    :param whence: where it's coming from
    :return: cropped array
    """
    # from numpy
    if whence == 'numpy/coronal':
        to_return = arr[len(arr) - 110 - 40:len(arr) - 110]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


# TODO: more experimentation on sagittal scan MIPing
def crop_normal_sagittal(arr: np.ndarray, whence: str):
    """
    Crops sagittal scans to be single-channel MIPed

    :param arr: sagittal array to be cropped
    :param whence: where it's coming from
    :return: cropped array
    """
    # from numpy
    if whence == 'numpy/sagittal':
        to_return = arr[len(arr) - 65 - 40:len(arr) - 65]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


def crop_multichannel_axial(arr: np.ndarray, whence: str):
    """
    Crop image to 3 channels of 25 slices to then be MIPed

    :param arr: array to be MIPed
    :param whence: where it's coming from
    :return: cropped array
    """
    num_slices = 3
    # from numpy
    if whence == 'numpy/axial':
        # Make a set of zeros of shape (3, 25, x, y)
        to_return = np.zeros((3, 25, len(arr[0]), len(arr[0][0])))
        chunk_start = 30
        chunk_end = chunk_start + 25
        inc = 25
    # from luke
    else:
        # Make a set of zeros of shape (3, 21, x, y)
        to_return = np.zeros((3, 21, len(arr[0]), len(arr[0][0])))
        chunk_start = 1
        chunk_end = chunk_start + 21
        inc = 21

    # Loop through and assign values to to_return
    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def crop_multichannel_axial_fa(arr: np.ndarray, whence: str):
    """
    Crop image to 3 channels of 25 slices to then be MIPed. Exactly the same
    as crop_multichannel_axial(), just starting 15 slices lower.

    :param arr: array to be MIPed
    :param whence: where it's coming from
    :return: cropped array
    """
    num_slices = 3
    # from numpy
    if whence == 'numpy/axial':
        to_return = np.zeros((3, 25, len(arr[0]), len(arr[0][0])))
        chunk_start = 45
        chunk_end = chunk_start + 25
        inc = 25
    # from luke
    else:
        to_return = np.zeros((3, 21, len(arr[0]), len(arr[0][0])))
        chunk_start = 1
        chunk_end = chunk_start + 21
        inc = 21
    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def crop_multichannel_coronal(arr: np.ndarray):
    """
    Crop coronal image to 3 channels of 30 slices to then be MIPed. Exactly the
    same as crop_multichannel_axial(), just from a different perspective. The
    source directory is assumed to be numpy/coronal.

    :param arr: array to be MIPed
    :return: cropped array
    """
    num_slices = 3
    to_return = np.zeros((3, 30, len(arr[0]), len(arr[0][0])))
    chunk_start = 70
    chunk_end = chunk_start + 30
    inc = 30
    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def crop_overlap_axial(arr: np.ndarray, whence: str):
    """
    Crop image to 20 channels of 25 slices (or 10 slices) to then be MIPed.
    Exactly the same as crop_multichannel_axial(), just with way more slices.

    :param arr: array to be MIPed
    :param whence: where it's coming from
    :return: cropped array
    """
    num_slices = 20
    # from numpy
    if whence == 'numpy/axial':
        to_return = np.zeros((num_slices, 25, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 15
        chunk_end = chunk_start + 25
        inc = 5
    # from luke
    else:
        to_return = np.zeros((num_slices, 10, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 4
        chunk_end = chunk_start + 10
        inc = 3

    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def crop_overlap_axial_fa(arr: np.ndarray, whence: str):
    """
    Crop image to 20 channels of 25 slices (or 10 slices) to then be MIPed.
    Exactly the same as crop_overlap_axial(), just starting lower.

    :param arr: array to be MIPed
    :param whence: where it's coming from
    :return: cropped array
    """
    num_slices = 20
    # from numpy
    if whence == 'numpy/axial':
        to_return = np.zeros((num_slices, 25, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 30
        chunk_end = chunk_start + 25
        inc = 5
    # from luke
    else:
        to_return = np.zeros((num_slices, 10, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 4
        chunk_end = chunk_start + 10
        inc = 3

    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def crop_overlap_coronal(arr: np.ndarray):
    """
    Crop image to 20 channels of 25 slices (or 10 slices) to then be MIPed.
    Exactly the same as crop_overlap_axial(), just from a different perspective
    The assumed source directory is numpy/coronal.

    :param arr: array to be MIPed
    :return: cropped array
    """
    num_slices = 20
    to_return = np.zeros((num_slices, 25, len(arr[0]), len(arr[0][0])))
    print(to_return.shape)
    chunk_start = 45
    chunk_end = chunk_start + 25
    inc = 5
    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def remove_extremes(arr: np.ndarray):
    """
    Removes extreme values from the array to be MIPed

    :param arr: array to be clipped
    :return: clipped array
    """
    # Filter out values greater than 270 and lower than 0
    a = arr > 270
    b = arr < 0
    arr[a] = -50
    arr[b] = -50
    return arr


def normalize(image, lower_bound=None, upper_bound=None):
    """
    Normalizes array -- not actually used in our processing pipeline

    :param image: image to be normalized
    :param lower_bound: lower bound to normalize w/r/t
    :param upper_bound: upper bound to normalize w/r/t
    :return: normalized image
    """
    if lower_bound is None:
        lower_bound = image.min()
    if upper_bound is None:
        upper_bound = image.max()

    image[image > upper_bound] = upper_bound
    image[image < lower_bound] = lower_bound

    return (image - image.mean()) / image.std()


def segment_vessels(arr: np.ndarray):
    """
    More extreme version of remove_extremes(). Taken from strip_skull.py.

    :param arr: Array to be segmented
    :return: segmented array
    """
    a = arr > 500
    b = arr < 120
    arr[a] = -50
    arr[b] = -50
    return arr
