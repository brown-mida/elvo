"""Methods for converting raw Dicom files in uncompressed .npz
files.
"""
import io
import logging
import os
import shutil
from typing import List

import pydicom
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
    stream = io.BytesIO()
    # noinspection PyTypeChecker
    np.savez(stream, arrays)
    processed_blob = storage.Blob(blob_name, bucket=bucket)
    stream.seek(0)
    processed_blob.upload_from_file(stream)
    logging.info(f'saving dicom data to GCS in {blob_name}')
    stream.close()


def process_patient(gcs_dir: str, bucket: storage.Bucket) -> List[np.ndarray]:
    """
    Prepares patient

    :param blob:
    :return:
    """
    if gcs_dir[-1] != '/':
        gcs_dir += '/'
    print(f'Processing GCS subdirectory: {gcs_dir}')

    os.makedirs('tmp', exist_ok=True)
    arrays = []
    for i in range(1, 4):
        prefix = gcs_dir + f'mip{i}/'
        mip_dir = os.path.join('tmp', gcs_dir)
        os.makedirs(mip_dir)
        blob: storage.Blob
        for j, blob in enumerate(bucket.list_blobs(prefix=prefix)):
            blob_filename = os.path.join(mip_dir, f'{j}.dcm')
            blob.download_to_filename(blob_filename)
        slices = load_scan(mip_dir)
        arr = preprocess_scan(slices)
        arrays.append(arr)
        shutil.rmtree('tmp/' + gcs_dir)
    return arrays


def prepare_multiphase():
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('data-elvo')
    blob: storage.Blob

    dirs = set()
    for blob in bucket.list_blobs(prefix='multiphase'):
        if ('multiphase/positive' in blob.name or
            'multiphase/negative' in blob.name):
            name_parts = blob.name.split('/')
            # in_dir = multiphase/(positive|negative)/<id>/
            in_dir = '/'.join(name_parts[0:3])
            dirs.add(in_dir)

    for in_dir in dirs:
        try:
            print(f'Processing dir {in_dir}')
            arrays = process_patient(in_dir, bucket)
            out_dir = 'airflow/' + in_dir
            savez_to_gcs(arrays, out_dir, bucket)
        except Exception:
            print(f"Error processing {in_dir}")