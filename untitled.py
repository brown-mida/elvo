# Code borrowed from
# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook

# TODO: Give this file a good name
# TODO: Create more clear instructions for running these files
import os
import re

import dicom
import numpy as np
import scipy.ndimage

from skimage.measure import marching_cubes


def load_scan(path):
    """Takes in the path of a directory containing scans and
    returns a list of dicom dataset objects. Each dicom dataset
    contains a single image slice.
    """
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]
                                 - slices[1].ImagePositionPatient[2])
    except KeyError:
        slice_thickness = np.abs(slices[0].SliceLocation
                                 - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    """Takes in a list of dicom datasets and returns the 3D pixel
    matrix, taking slope and intercept into account.
    """
    image = np.stack([np.frombuffer(s.PixelData, np.int16).reshape(512, 512)
                      for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, slices, new_spacing=[1, 1, 1]):
    """Takes in a 3D image, a list of slices and resizes the image so
    each pixel corresponds to the same real-world dimensions.
    """
    # Determine current pixel spacing
    spacing = np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


if __name__ == '__main__':
    DATA_DIR = 'RI Hospital ELVO Data'  # The relative path to the dataset

    id_pattern = re.compile(r'\d+')
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        if filenames and '.dcm' in filenames[0]:
            slices = load_scan(dirpath)
            pixels = get_pixels_hu(slices)
            pixels_resampled, spacing = resample(pixels, slices)
            # TODO: Zero-centering, normalization
            # TODO: Show plots of the images in a Jupyter notebook
            # TODO: Not all of the 3D representations are of the same height
            # We can just crop the lower body in the future
            outfile = 'patient-{}'.format(id_pattern.findall(dirpath)[0])
            np.save(outfile, pixels_resampled)
