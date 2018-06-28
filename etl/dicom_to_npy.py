"""To run this GOOGLE_APPLICATION_CREDENTIALS should be set.

This script converts the compressed files containing dicom files
to numpy files.
"""
import io
import logging
import os
import shutil
import subprocess
import time
from typing import List

import numpy as np
import pydicom
from google.cloud import storage

from lib import parsers
from lib import transforms


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def process_cab(blob: storage.Blob, patient_id: str) -> np.ndarray:
    """
    Downloads the blob and return a 3D standardized numpy array.

    :param blob:
    :param patient_id:
    :return:
    """
    # TODO: Fix issues with process_cab workingdir failing
    old_wd = os.getcwd()
    dirname = f'/tmp/dicom_to_npy-{int(time.time())}'
    os.makedirs(dirname, exist_ok=True)
    os.chdir(dirname)

    blob.download_to_filename(patient_id + '.cab')
    subprocess.call(['cabextract', patient_id + '.cab'])
    logging.info('extracted cab file')

    dirpath = list(os.walk('.'))[2][0]
    logging.info(f'loading scans from {dirpath}')
    processed_scan = _process_cab(dirpath)

    os.chdir(old_wd)
    shutil.rmtree(dirname)
    return processed_scan


def _process_cab(dirpath: str) -> np.array:
    scan = parsers.load_scan(dirpath)
    processed_scan = preprocess_scan(scan)
    return processed_scan


def process_zip(blob: storage.Blob, patient_id: str) -> np.ndarray:
    """
    Downloads the blob and returns a 3D standardized numpy array.
    :param blob:
    :param patient_id:
    :return:
    """
    old_wd = os.getcwd()
    dirname = f'/tmp/dicom_to_npy-{int(time.time())}'
    os.makedirs(dirname, exist_ok=True)
    os.chdir(dirname)

    blob.download_to_filename(patient_id + '.zip')
    logging.info('extracting zip file')
    shutil.unpack_archive(patient_id + '.zip', format='zip')

    dirpath = list(os.walk('.'))[3][0]
    logging.info(f'loading scans from {dirpath}')
    scan = parsers.load_scan(dirpath)
    processed_scan = preprocess_scan(scan)
    logging.info(f'processing dicom data')

    os.chdir(old_wd)
    shutil.rmtree(dirname)
    return processed_scan


def save_to_gcs(processed_scan, outpath, bucket):
    stream = io.BytesIO()
    np.save(stream, processed_scan)
    processed_blob = storage.Blob(outpath, bucket=bucket)
    stream.seek(0)
    processed_blob.upload_from_file(stream)
    logging.info(f'saving dicom data to GCS in {outpath}')


def preprocess_scan(slices: List[pydicom.FileDataset]) -> np.array:
    """Transforms the input dicom slices into a numpy array of pixels
    in Hounsfield units with standardized spacing.
    """
    scan = transforms.get_pixels_hu(slices)
    scan = transforms.standardize_spacing(scan, slices)
    return scan


def dicom_to_npy(in_dir, out_dir):
    """
    :param in_dir: directory in gs://elvos to load from. must end with /
    :param out_dir: directory in gs://elvos to save to. must end with /
    :return:
    """
    gcs_client = storage.Client(project='elvo-198322')
    bucket = gcs_client.get_bucket('elvos')

    blob: storage.Blob
    for blob in bucket.list_blobs(prefix=in_dir):
        if len(blob.name) < 4 or blob.name[-4:] not in ('.zip', '.cab'):
            logging.info(f'ignoring non-data file {blob.name}')
            continue

        logging.info(f'processing blob {blob.name}')
        patient_id = blob.name[len(in_dir): -len('.cab')]
        outpath = f'{out_dir}{patient_id}.npy'

        if storage.Blob(outpath, bucket).exists():
            logging.info(f'outfile {outpath} already exists')
            continue
        elif blob.name.endswith('.cab'):
            processed_scan = process_cab(blob, patient_id)
            save_to_gcs(processed_scan, outpath, bucket)
        elif blob.name.endswith('.zip'):
            processed_scan = process_zip(blob, patient_id)
            save_to_gcs(processed_scan, outpath, bucket)
        else:
            logging.info(f'file extension must be .cab or .zip,'
                         f' got {blob.name}')


if __name__ == '__main__':
    configure_logger()
    dicom_to_npy('ELVOs_anon/', 'raw_numpy/')
