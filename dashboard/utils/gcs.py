"""
Google Cloud Storage related logic.
"""

import io
import numpy as np
from google.cloud import storage


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


def upload_npy_to_gcs(arr, filename, user, bucket):
    np.save('../tmp/{}.npy'.format(filename), arr)
    out_blob = bucket.blob('{}/files/{}.npy'.format(user, filename))
    out_blob.upload_from_filename('../tmp/{}.npy'.format(filename))
