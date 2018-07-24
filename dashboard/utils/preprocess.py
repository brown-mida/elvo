"""To run this GOOGLE_APPLICATION_CREDENTIALS should be set.

This script converts the compressed files containing dicom files
to numpy files.
"""
import logging
import numpy as np
import os
import pydicom
import shutil
import subprocess
from typing import List

from utils.transforms import get_pixels_hu, standardize_spacing


def process_cab(file, filename, tmp_dir):
    """
    Downloads the blob and return a 3D standardized numpy array.

    :param blob:
    :param patient_id:
    :return:
    """
    # TODO: Fix issues with process_cab workingdir failing
    current_dir = os.getcwd()
    file.save(os.path.join(tmp_dir, filename))
    os.chdir(tmp_dir)

    subprocess.call(['cabextract', filename], stdout=open(os.devnull, 'wb'))
    logging.info('extracted cab file')

    dcm_path = list(os.walk('.'))[2][0]
    logging.info('Loading scans from {}'.format(dcm_path))
    processed_scan = _process_cab(dcm_path)
    shutil.rmtree(list(os.walk('.'))[1][0])
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
    scan = get_pixels_hu(slices)
    scan = standardize_spacing(scan, slices)
    return scan
