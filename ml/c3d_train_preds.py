import csv
import io
import logging
import os
import random
from models.three_d import c3d
import numpy as np
import tensorflow as tf
from google.cloud import storage
from tensorflow.python.lib.io import file_io

BLACKLIST = []
LEARN_RATE = 1e-5


def download_array(blob: storage.Blob) -> np.ndarray:
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def save_preds_to_cloud(arr: np.ndarray, type: str, id: str):
    """Uploads chunk .npy files to gs://elvos/chunk_data/<patient_id>.npy
    """
    try:
        print(f'gs://elvos/chunk_data/preds/{id}.npy')
        np.save(file_io.FileIO(f'gs://elvos/chunk_data/preds/{type}/{id}.npy', 'w'),
                arr)
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    train_ids = {}
    val_ids = {}
    with open('train_ids.csv', 'r') as pos_file:
        reader = csv.reader(pos_file, delimiter=',')
        for row in reader:
            print(row)
            train_ids[row[0]] = ''

    with open('val_ids.csv', 'r') as pos_file:
        reader = csv.reader(pos_file, delimiter=',')
        for row in reader:
            print(row)
            train_ids[row[0]] = ''

    print(train_ids)
    print(val_ids)

    # Get npy files from Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        # '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        'credentials/client_secret.json'
    )
    bucket = gcs_client.get_bucket('elvos')

    model = c3d.C3DBuilder.build()
    model.load_weights('tmp/c3d_separated_ids.hdf5')

    for blob in bucket.list_blobs(prefix='airflow/npy'):

        arr = download_array(blob)
        chunks = []

        for i in range(0, len(arr), 32):
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    chunk = arr[i: i + 32,
                                j: j + 32,
                                k: k + 32]
                    chunks.append(chunk)

        preds = model.predict(chunks, batch_size=16)
        print(preds)

        train = False
        val = False
        file_id = blob.name.split('/')[-1].split('.')[0][:16]
        print(file_id)

        if file_id in train_ids:
            train = True

        elif file_id in val_ids:
            val = True

        else:
            rand = random.randint(1, 100)
            if rand > 10:
                train = True
            else:
                val = True

        if train:
            save_preds_to_cloud(preds, 'train', file_id)

        if val:
            save_preds_to_cloud(preds, 'val', file_id)


if __name__ == '__main__':
    main()
