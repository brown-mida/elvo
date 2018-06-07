"""Downloads the data from gs://elvos/numpy to thingumy.

The data will be outputted to /home/lzhu7/data/numpy/
"""
import logging
import os
import subprocess

from google.cloud import storage

if __name__ == '__main__':
    client = storage.Client(project='elvo-198322')
    bucket = storage.Bucket(client, name='elvos')

    # TODO: Remove duplication of logger code
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    blob: storage.Blob
    for blob in bucket.list_blobs(prefix='numpy/'):
        try:
            filename = blob.name[len('numpy/'):]
            logging.info(f'downloading {filename}')
            blob.download_to_filename(filename)
            subprocess.call(['scp',
                             filename,
                             'thingumy:/home/lzhu7/data/numpy'])
            os.remove(filename)
        except Exception as e:
            logging.error(e)
