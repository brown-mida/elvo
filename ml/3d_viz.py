import csv
import io
import logging
import os
import random
from matplotlib import pyplot as plt
import math
from models.three_d import c3d
import tensorflow as tf
from google.cloud import storage
import numpy as np


def download_array(blob: storage.Blob) -> np.ndarray:
    """Downloads data blobs as numpy arrays

    :param blob: the GCS blob you want to download as an array
    :return:
    """
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def display_preds(preds: np.ndarray):
    return


def make_preds():
    # Get npy files from Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        # Use this when running on VM
        '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        # Use this when running locally
        # 'credentials/client_secret.json'
    )
    bucket = gcs_client.get_bucket('elvos')

    # load model
    model = c3d.C3DBuilder.build()
    model.load_weights('tmp/FINAL_RUN_6.hdf5')

    # Get every scan in airflow/npy
    for blob in bucket.list_blobs(prefix='airflow/test_npy'):
        arr = download_array(blob)

        # Get every chunk in the scan
        preds = []
        for i in range(0, len(arr), 32):
            layer = np.zeros(shape=(int(math.ceil(len(arr[0] / 32)) * math.ceil(len(arr[0][0] / 32)))))
            layer_idx = 0
            print(layer.shape)
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    chunk = arr[i: i + 32,
                                j: j + 32,
                                k: k + 32]
                    airspace = np.where(chunk < -300)
                    # if it's less than 90% airspace
                    if (airspace[0].size / chunk.size) < 0.9:
                        # append it to a list of chunks for this brain
                        if chunk.shape == (32, 32, 32):
                            chunk = np.expand_dims(chunk, axis=-1)
                            layer[layer_idx] = model.predict(chunk)
                    layer_idx += 1
            layer = layer.reshape((int(math.ceil(len(arr[0] / 32))), int(math.ceil(len(arr[0][0] / 32)))))
            preds.append(layer)

        preds = np.asarray(preds)
        print(arr.shape)
        print(preds.shape)


def main():
    tf.Session(config=tf.ConfigProto(log_device_placement=True))
    make_preds()


if __name__ == '__main__':
    main()
