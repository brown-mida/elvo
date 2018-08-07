"""
This script uses a trained C3D model to make predictions about all of the
chunks in a brain scan. This vector of predictions is then meant to be fed
into another simple NN to classify it as either R/L MCA, R/L ICA, basilar,
or R/L vertebral.
"""

import csv
import io
import logging
import os
import random
from ml.models.three_d import c3d
import numpy as np
import tensorflow as tf
from google.cloud import storage
from tensorflow.python.lib.io import file_io
from ensemble_best_3d import get_ensembles

BLACKLIST = []
LEARN_RATE = 1e-5


def download_array(blob: storage.Blob) -> np.ndarray:
    """Downloads data blobs as numpy arrays

    :param blob: the GCS blob you want to download as an array
    :return:
    """
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def save_preds_to_cloud(arr: np.ndarray, type: str, id: str):
    """Uploads chunk .npy files to gs://elvos/chunk_data/preds/<type>/<id>.npy

    :param arr: the numpy array to upload
    :param type: train, val, or test — what dataset the array is in
    :param id: the ID of the scan
    :return:
    """
    try:
        print(f'gs://elvos/chunk_data/preds/{type}/{id}.npy')
        np.save(file_io.FileIO(f'gs://elvos/chunk_data/preds/{type}/{id}.npy',
                               'w'), arr)
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')


def main():
    # Make sure GPU is being used
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Load training set IDs, validation set IDs, and testing set IDs
    train_ids = {}
    val_ids = {}
    test_ids = {}
    with open('train_ids.csv', 'r') as pos_file:
        reader = csv.reader(pos_file, delimiter=',')
        for row in reader:
            if row[1] != '0':
                train_ids[row[0]] = ''
    with open('val_ids.csv', 'r') as pos_file:
        reader = csv.reader(pos_file, delimiter=',')
        for row in reader:
            if row[1] != '0':
                val_ids[row[0]] = ''
    with open('test_ids.csv', 'r') as pos_file:
        reader = csv.reader(pos_file, delimiter=',')
        for row in reader:
            if row[1] != '0':
                test_ids[row[0]] = ''

    # Get npy files from Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        # Use this when running on VM
        '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        # Use this when running locally
        # 'credentials/client_secret.json'
    )
    bucket = gcs_client.get_bucket('elvos')

    # load model
    models = get_ensembles()

    # Get every scan in airflow/npy
    for blob in bucket.list_blobs(prefix='airflow/npy'):
        arr = download_array(blob)

        # Get every chunk in the scan
        preds = []
        for i in range(0, len(arr), 32):
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
                            chunk = np.expand_dims(chunk, axis=0)
                            chunk = np.expand_dims(chunk, axis=-1)
                            pred = models[0].predict(chunk)
                            preds.append(pred)
                    else:
                        preds.append(0)

        # Use the model to predict about every chunk in this brain
        preds = np.asarray(preds)
        print(preds.shape)

        # Figure out which dataset this ID is in
        file_id = blob.name.split('/')[-1].split('.')[0][:16]
        train = False
        val = False
        test = False
        if file_id in train_ids:
            train = True
        elif file_id in val_ids:
            val = True
        elif file_id in test_ids:
            test = True

        # If it's in none of them, randomly figure out which one it's in
        else:
            rand = random.randint(1, 100)
            if rand > 10:
                train = True
            else:
                val = True

        # Upload to GCS
        if train:
            save_preds_to_cloud(preds, 'train', file_id)
        if val:
            save_preds_to_cloud(preds, 'val', file_id)
        if test:
            save_preds_to_cloud(preds, 'test', file_id)


if __name__ == '__main__':
    main()
