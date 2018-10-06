"""
Script used to compress numpy files to .npz files so that they can be saved on
department GPU 1708 and then trained upon.
"""

import io
import logging
from typing import Dict

import numpy as np
from google.cloud import storage


def save_arrays(arrays: Dict[str, np.ndarray],
                filename: str,
                bucket: storage.Bucket):
    """
    Saves .npz arrays (compressed in groups of 10) to cloud

    :param arrays: dict mapping IDs --> arrays to be saved
    :param filename: new filename
    :param bucket: bucket to be saved within
    :return:
    """
    out_stream = io.BytesIO()
    np.savez_compressed(out_stream, **arrays)
    out_stream.seek(0)
    out_blob = bucket.blob(filename)
    out_blob.upload_from_file(out_stream)


def compress_numpy(in_dir, out_dir):
    """
    Saves the arrays in batches of 10.

    :param in_dir: directory to be sourcing the compression from
    :param out_dir: directory to be sourcing the compression to
    :return:
    """
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('elvos')

    blob: storage.Blob
    arrays = {}
    i = 0
    # for every blob in the in_dir
    for blob in bucket.list_blobs(prefix=in_dir):
        patient_id = blob.name[len(in_dir): -len('.npy')]
        # Case >= 10 arrays left to be uploaded
        if len(arrays) >= 10:
            # Upload arrays
            logging.info(f'uploading arrays: {list(arrays.keys())}')
            save_arrays(arrays,
                        f'{out_dir}{i}.npz',
                        bucket)
            arrays = {}
            i += 1
        # add array to dict
        in_stream = io.BytesIO()
        logging.info(f'downloading {blob.name}, patient id: {patient_id}')
        blob.download_to_file(in_stream)
        in_stream.seek(0)
        arr = np.load(in_stream)
        arrays[patient_id] = arr

    # Upload remaining files
    if len(arrays) > 0:
        logging.info(f'uploading arrays: {list(arrays.keys())}')
        save_arrays(arrays,
                    f'{out_dir}{i}.npz',
                    bucket)
