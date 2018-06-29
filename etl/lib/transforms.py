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


def mip_normal(array: np.ndarray) -> np.ndarray:
    return np.max(array, axis=0)


def mip_multichannel(array: np.ndarray) -> np.ndarray:
    num_slices = 3
    to_return = np.zeros((num_slices, len(array[0][0]), len(array[0][0][0])))
    print(to_return.shape)
    for i in range(num_slices):
        print(array[i].shape)
        to_return[i] = np.max(array[i], axis=0)
    return to_return


def mip_overlap(array: np.ndarray) -> np.ndarray:
    num_slices = 20
    to_return = np.zeros((num_slices, len(array[0][0]), len(array[0][0][0])))
    print(to_return.shape)
    for i in range(num_slices):
        print(array[i].shape)
        to_return[i] = np.max(array[i], axis=0)
    return to_return


def crop_normal_axial(arr: np.ndarray, whence: str):
    # from numpy
    if whence == 'numpy/axial':
        to_return = arr[len(arr) - 35 - 64:len(arr) - 35]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


def crop_strip_skull(arr: np.ndarray, whence: str):
    # from numpy
    if whence == 'numpy/axial':
        initial_slice_num = len(arr) - 70
        to_return = arr[initial_slice_num - 24:initial_slice_num]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


def crop_normal_coronal(arr: np.ndarray, whence: str):
    print(arr.shape)
    # from numpy
    if whence == 'numpy/coronal':
        to_return = arr[len(arr) - 110 - 40:len(arr) - 110]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


# TODO: more experimentation on sagittal scan MIPing
def crop_normal_sagittal(arr: np.ndarray, whence: str):
    print(arr.shape)
    # from numpy
    if whence == 'numpy/sagittal':
        to_return = arr[len(arr) - 65 - 40:len(arr) - 65]
    # from luke
    else:
        to_return = arr[len(arr) - 40:]
    return to_return


def crop_multichannel_axial(arr: np.ndarray, whence: str):
    num_slices = 3
    # from numpy
    if whence == 'numpy/axial':
        to_return = np.zeros((3, 25, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 30
        chunk_end = chunk_start + 25
        inc = 25
        # from luke
    else:
        to_return = np.zeros((3, 21, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 1
        chunk_end = chunk_start + 21
        inc = 21
    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def crop_multichannel_coronal(arr: np.ndarray, whence: str):
    num_slices = 3
    # from numpy
    if whence == 'numpy/coronal':
        to_return = np.zeros((3, 30, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 70
        chunk_end = chunk_start + 30
        inc = 30
        # from luke
    else:
        to_return = np.zeros((3, 21, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 1
        chunk_end = chunk_start + 21
        inc = 21
    for i in range(num_slices):
        to_return[i] = arr[len(arr) - chunk_end:len(arr) - chunk_start]
        chunk_start += inc
        chunk_end += inc
    return to_return


def crop_overlap_axial(arr: np.ndarray, whence: str):
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


def crop_overlap_coronal(arr: np.ndarray, whence: str):
    num_slices = 20
    # from numpy
    if whence == 'numpy/coronal':
        to_return = np.zeros((num_slices, 25, len(arr[0]), len(arr[0][0])))
        print(to_return.shape)
        chunk_start = 45
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


def remove_extremes(arr: np.ndarray):
    a = arr > 270
    b = arr < 0
    arr[a] = -50
    arr[b] = -50
    return arr


def normalize(image, lower_bound=None, upper_bound=None):
    if lower_bound is None:
        lower_bound = image.min()
    if upper_bound is None:
        upper_bound = image.max()

    image[image > upper_bound] = upper_bound
    image[image < lower_bound] = lower_bound

    return (image - image.mean()) / image.std()


def segment_vessels(arr: np.ndarray, whence: str):
    # from numpy
    if whence == 'numpy/axial':
        a = arr > 520
        b = arr < 120
        arr[a] = -50
        arr[b] = -50
        return arr
    # from luke
    # else:
    #     to_return = arr[len(arr) - 40:]
