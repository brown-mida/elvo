"""Methods for converting raw Dicom files in uncompressed .npz
files.
"""
import io
import logging
import os
import shutil
import traceback
from typing import List

import numpy as np
from google.cloud import storage

from .prepare_arrays import load_scan, preprocess_scan


def savez_to_gcs(arrays: List[np.ndarray],
                 blob_name: str,
                 bucket: storage.Bucket) -> None:
    """
    Saves the array to GCS.

    :param array: the numpy array to save
    :param blob_name: the name of the output blob
    :param bucket: the output bucket
    :return:
    """
    logging.info(f'saving data to GCS blob: {blob_name}')

    stream = io.BytesIO()
    # noinspection PyTypeChecker
    np.savez(stream, *arrays)
    processed_blob = storage.Blob(blob_name, bucket=bucket)
    stream.seek(0)
    processed_blob.upload_from_file(stream)
    stream.close()


def process_patient(gcs_dir: str,
                    bucket: storage.Bucket) -> List[np.ndarray]:
    """
    Prepares patient

    :param blob:
    :return:
    """
    if gcs_dir[-1] != '/':
        gcs_dir += '/'
    logging.info(f'processing GCS subdirectory: {gcs_dir}')

    os.makedirs('multiphase', exist_ok=True)
    arrays = []
    for i in range(1, 4):
        logging.debug(f'downloading dicom slices from GCS')

        prefix = os.path.join(gcs_dir + f'mip{i}')
        mip_dir = os.path.join('tmp', prefix)
        shutil.rmtree(mip_dir, ignore_errors=True)
        os.makedirs(mip_dir)
        blob: storage.Blob
        for j, blob in enumerate(bucket.list_blobs(prefix=prefix)):
            blob_filename = os.path.join(mip_dir, f'{j}.dcm')
            blob.download_to_filename(blob_filename)

        logging.debug(f'loading and processing slices')
        slices = load_scan(mip_dir)
        arr = preprocess_scan(slices)
        arrays.append(arr)

    shutil.rmtree('multiphase', ignore_errors=True)
    return arrays


def prepare_multiphase():
    """
    Loads data from gs://multiphase/positives and gs://multiphase/negatives
    and saves the files
    """
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('data-elvo')
    blob: storage.Blob

    dirs = set()

    for blob in bucket.list_blobs(prefix='multiphase'):
        if ('multiphase/positive' in blob.name
                or 'multiphase/negative' in blob.name):
            name_parts = blob.name.split('/')
            # in_dir is multiphase/(positive|negative)/<id>/
            in_dir = '/'.join(name_parts[0:3])
            dirs.add(in_dir)

    for in_dir in dirs:
        try:
            arrays = process_patient(in_dir, bucket)
            out_dir = 'airflow/' + in_dir
            savez_to_gcs(arrays, out_dir, bucket)
        except Exception as e:
            logging.error(f"error processing {in_dir}")
            logging.error(traceback.format_exc())
