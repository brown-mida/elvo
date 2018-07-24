"""
Google Cloud Storage related logic.
"""

import io
import os
import numpy as np
from google.cloud import storage
import scipy.misc


def authenticate():
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
