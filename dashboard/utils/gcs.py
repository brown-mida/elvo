"""
Google Cloud Storage related logic.
"""

import logging

import io
import numpy as np
import os
import scipy.misc
from google.cloud import storage


def authenticate():
    # TODO(luke): This won't work on app engine
    return storage.Client.from_service_account_json(
        # for running on the airflow GPU
        # '/home/lukezhu/elvo-analysis/credentials/client_secret.json'

        # for running locally
        '../credentials/client_secret.json'
    )


def upload_to_gcs(file, filename, bucket):
    buf = io.BufferedRandom(io.BytesIO())
    buf.write(file.stream.read())
    buf.seek(0)
    out_blob = storage.Blob(filename, bucket)
    out_blob.upload_from_file(buf)


def upload_npy_to_gcs(arr, filename, user, dataset, bucket):
    np.save('../tmp/{}.npy'.format(filename), arr)
    out_blob = bucket.blob('{}/{}/{}.npy'.format(user, dataset, filename))
    out_blob.upload_from_filename('../tmp/{}.npy'.format(filename))
    os.remove('../tmp/{}.npy'.format(filename))


def save_npy_as_image_and_upload(arr, user, dataset, folder,
                                 filename, bucket, tmp_dir):
    scipy.misc.imsave('{}/{}.jpg'.format(tmp_dir, filename), arr)
    out_blob = bucket.blob('{}/{}/{}/{}.jpg'.format(user, dataset,
                                                    folder, filename))
    out_blob.upload_from_filename('{}/{}.jpg'.format(tmp_dir, filename))
    os.remove('{}/{}.jpg'.format(tmp_dir, filename))


def download_image(user, dataset, data_type, filename, bucket):
    logging.info("A")
    blob = bucket.blob('{}/{}/{}/{}.jpg'.format(user, dataset,
                                                data_type, filename))
    out_stream = io.BytesIO()
    blob.download_to_file(out_stream)
    out_stream.seek(0)
    return out_stream
