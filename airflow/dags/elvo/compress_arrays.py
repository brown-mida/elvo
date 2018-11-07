"""
Methods for compressing .npy files into .npz files.

These are needed because of limited disk space
on GPU 1708.
"""

import io
import logging
from typing import Dict

import numpy as np
from google.cloud import storage


def _upload_npz(arrays: Dict[str, np.ndarray],
                filename: str,
                bucket: storage.Bucket) -> None:
    """
    Saves the arrays to GCS in compressed .npz format.

    :param arrays: a dict mapping IDs to arrays
    :param filename: a GCS blob name
    :param bucket: a GCS bucket name
    :return:
    """
    out_stream = io.BytesIO()
    np.savez_compressed(out_stream, **arrays)
    out_stream.seek(0)
    out_blob = bucket.blob(filename)
    out_blob.upload_from_file(out_stream)


def compress_arrays(in_dir: str, out_dir: str) -> None:
    """
    Saves .npy files as .npz files.

    Each .npz file contain the contents of 10 .npy files with patient IDs
    for keys. See the numpy documentation for info on how
    to load these files.

    :param in_dir: directory with .npy files within the 'elvos' bucket,
        ending with a '/'
    :param out_dir: directory to save compressed .npz files in
        within the 'elvos' bucket, ending with a '/'
    :return:
    """
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('data-elvo')

    blob: storage.Blob
    arrays = {}
    i = 0
    for blob in bucket.list_blobs(prefix=in_dir):
        patient_id = blob.name[len(in_dir): -len('.npy')]
        if len(arrays) == 10:
            logging.info(f'uploading arrays: {list(arrays.keys())}')
            _upload_npz(arrays,
                        f'{out_dir}{i}.npz',
                        bucket)
            arrays = {}
            i += 1
        in_stream = io.BytesIO()
        logging.info(f'downloading {blob.name}, patient id: {patient_id}')
        blob.download_to_file(in_stream)
        in_stream.seek(0)
        arr = np.load(in_stream)
        arrays[patient_id] = arr

    # Upload the remaining files
    if len(arrays) > 0:
        logging.info(f'uploading arrays: {list(arrays.keys())}')
        _upload_npz(arrays,
                    f'{out_dir}{i}.npz',
                    bucket)
