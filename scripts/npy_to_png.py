"""Converts the .npy files in gcs://elvos/numpy/ to
a directory of axial-view PNGs in gcs://elvos/png/.

If scipy.misc does not work, be sure to install Pillow==5.1.0
"""
import io
import logging

import numpy as np
from google.cloud import storage
from scipy import misc


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


def upload_array(arr: np.ndarray, dirname: str, bucket: storage.Bucket):
    """Uploads axial slice PNGs to gs://elvos/<DIRNAME>/axial/.
    """
    for i in range(len(arr)):
        out_stream = io.BytesIO()
        misc.imsave(out_stream, arr[i], format='png')
        out_filename = f'{dirname}/axial/{i:05d}.png'
        out_blob = storage.Blob(out_filename, bucket)
        out_stream.seek(0)
        out_blob.upload_from_file(out_stream)


if __name__ == '__main__':
    configure_logger()
    client = storage.Client(project='elvo-198322')
    bucket = storage.Bucket(client, name='elvos')

    in_blob: storage.Blob
    for in_blob in bucket.list_blobs(prefix='numpy/'):
        logging.info(f'downloading {in_blob.name}')
        arr = download_array(in_blob)

        image_dirname = in_blob.name.replace('numpy/', 'png/')[:-len('.npy')]

        logging.info(f'uploading {image_dirname}')
        upload_array(arr, image_dirname, bucket)
