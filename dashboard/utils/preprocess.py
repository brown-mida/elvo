"""To run this GOOGLE_APPLICATION_CREDENTIALS should be set.

This script converts the compressed files containing dicom files
to numpy files.
"""
import logging
import subprocess
from typing import List

import numpy as np
import os
import pydicom
import shutil

from utils.gcs import save_npy_as_image_and_upload
import utils.transforms as t


def process_cab_from_file(file, filename, tmp_dir):
    """
    Downloads the blob and return a 3D standardized numpy array.

    :param blob:
    :param patient_id:
    :return:
    """
    # TODO: Fix issues with process_cab workingdir failing
    file.save(os.path.join(tmp_dir, filename))
    return process_cab(filename, tmp_dir)


def process_cab(filename, tmp_dir):
    """
    Downloads the blob and return a 3D standardized numpy array.

    :param blob:
    :param patient_id:
    :return:
    """
    # TODO: Fix issues with process_cab workingdir failing
    current_dir = os.getcwd()
    os.chdir(tmp_dir)

    subprocess.call(['cabextract', filename], stdout=open(os.devnull, 'wb'))
    logging.info('extracted cab file')

    dcm_path = list(os.walk('.'))[2][0]
    logging.info('Loading scans from {}'.format(dcm_path))
    processed_scan = _process_cab(dcm_path)
    shutil.rmtree(list(os.walk('.'))[1][0])
    os.remove(filename)
    os.chdir(current_dir)
    return processed_scan


def _process_cab(dirpath):
    scan = load_scan(dirpath)
    processed_scan = preprocess_scan(scan)
    return processed_scan


def load_scan(dirpath: str) -> List[pydicom.FileDataset]:
    """Loads a CT scan.

    This only loads dicom files ending in .dcm and ignores nested
    folders.

    :param dirpath: the path to a directory containing dicom (.dcm) files
    :return: a list of all dicom FileDataset objects, sorted by
        ImagePositionPatient
    """
    slices = [pydicom.read_file(dirpath + '/' + filename)
              for filename in os.listdir(dirpath)
              if filename.endswith('.dcm')]
    return sorted(slices, key=lambda x: float(x.ImagePositionPatient[2]))


def preprocess_scan(slices: List[pydicom.FileDataset]) -> np.array:
    """Transforms the input dicom slices into a numpy array of pixels
    in Hounsfield units with standardized spacing.
    """
    scan = t.get_pixels_hu(slices)
    scan = t.standardize_spacing(scan, slices)
    return scan


def generate_images(arr, user, dataset, filename, bucket, tmp_dir):
    axial = arr.max(axis=0)
    coronal = arr.max(axis=1)
    sagittal = arr.max(axis=2)
    save_npy_as_image_and_upload(axial, user, dataset, 'axial',
                                 filename, bucket, tmp_dir)
    save_npy_as_image_and_upload(coronal, user, dataset, 'coronal',
                                 filename, bucket, tmp_dir)
    save_npy_as_image_and_upload(sagittal, user, dataset, 'sagittal',
                                 filename, bucket, tmp_dir)


def generate_mip_images(arr, user, dataset, filename, bucket, tmp_dir):
    save_npy_as_image_and_upload(arr, user, dataset, 'mip',
                                 filename, bucket, tmp_dir)


def transform_array(arr, params):
    if params['flipZ']:
        arr = np.flip(arr, axis=0)

    if params['cropZ']:
        crop_min = int(float(params['cropZmin']))
        crop_max = int(float(params['cropZmax']))
        arr = t.crop_z(arr, crop_min, crop_max)

    if params['centerCropXY']:
        crop_size = int(float(params['centerCropSize']))
        arr = t.center_crop_xy(arr, crop_size)

    if params['boundHu']:
        bound_min = int(float(params['boundHuMin']))
        bound_max = int(float(params['boundHuMax']))
        arr = t.bound_hu(arr, bound_min, bound_max)

    if params['mip']:
        if params['multichannelMip']:
            arr = t.mip_multichannel(arr)
        else:
            arr = t.mip_normal(arr)
    return arr
