"""Converts the .npy files in gcs://elvos/numpy/ to
a directory of axial-view PNGs in gcs://elvos/png/.

If scipy.misc does not work, be sure to install Pillow==5.1.0
"""
import io
import logging

import numpy as np
from google.cloud import storage
from mayavi import mlab


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def download_array(blob: storage.Blob) -> np.ndarray:
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


if __name__ == '__main__':
    configure_logger()
    client = storage.Client(project='elvo-198322')
    bucket = storage.Bucket(client, name='elvos')

    in_blob: storage.Blob
    for in_blob in bucket.list_blobs(prefix='processed/luke/'):
        logging.info(f'downloading {in_blob.name}')
        arr = download_array(in_blob)
        mlab.contour3d(arr)
        mlab.show()
