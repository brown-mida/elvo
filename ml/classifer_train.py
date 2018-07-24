import tensorflow as tf
from models.three_d import cube_classifier
from keras.optimizers import Adadelta, SGD
from keras.losses import categorical_crossentropy
import pandas as pd
import numpy as np
from google.cloud import storage
import io
import os

BLACKLIST = []
LEARN_RATE = 1e-4


def download_array(blob: storage.Blob) -> np.ndarray:
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def train(x_train, y_train, x_val, y_val):
    model = cube_classifier.CubeClassifierBuilder.build()
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=16,
                        epochs=50,
                        validation_data=(x_val, y_val))

    model.evaluate(x_val, y_val, verbose=1)
    return history


def load_probs(labels: pd.DataFrame):
    # Get pred files from Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        # '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        'credentials/client_secret.json'
    )

    labels.set_index('Unnamed: 0', inplace=True)
    bucket = gcs_client.get_bucket('elvos')

    x_train = []
    y_train = []
    idx = 0
    for blob in bucket.list_blobs(prefix='chunk_data/preds/train'):
        if idx % 100 == 0:
            print(f'successfully loaded {idx} training scans and labels')
        file_id = blob.name.split('/')[-1].split('.')[0]
        arr = download_array(blob)
        diff = 784 - arr.shape[0]
        if diff > 0 and file_id in labels.index.values:
            arr = arr.tolist()
            for i in range(diff):
                arr.append([0])
            arr = np.asarray(arr)
            arr = np.reshape(arr, (28, 28, 1))
            x_train.append(arr)

            label = np.array([
                labels.loc[file_id]['L MCA'],
                labels.loc[file_id]['R MCA'],
                labels.loc[file_id]['L ICA'],
                labels.loc[file_id]['R ICA'],
                labels.loc[file_id]['L Vert'],
                labels.loc[file_id]['R Vert'],
                labels.loc[file_id]['Basilar'],
            ])
            label = label.astype(int)
            y_train.append(label)
            idx += 1

    x_val = []
    y_val = []
    idx = 0
    for blob in bucket.list_blobs(prefix='chunk_data/preds/val'):
        if idx % 100 == 0:
            print(f'successfully loaded {idx} validation scans and labels')
        file_id = blob.name.split('/')[-1].split('.')[0]
        arr = download_array(blob)
        diff = 784 - arr.shape[0]
        if diff > 0 and file_id in labels.index.values:
            arr = arr.tolist()
            for i in range(diff):
                arr.append([0])
            arr = np.asarray(arr)
            arr = np.reshape(arr, (28, 28, 1))
            x_val.append(arr)

            label = np.array([
                labels.loc[file_id]['L MCA'],
                labels.loc[file_id]['R MCA'],
                labels.loc[file_id]['L ICA'],
                labels.loc[file_id]['R ICA'],
                labels.loc[file_id]['L Vert'],
                labels.loc[file_id]['R Vert'],
                labels.loc[file_id]['Basilar'],
            ])
            label = label.astype(int)
            y_val.append(label)
            idx += 1
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    return x_train, y_train, x_val, y_val


def load_labels():
    return pd.read_csv('classification_vectors.csv')


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.Session(config=tf.ConfigProto(log_device_placement=True))

    labels = load_labels()
    x_train, y_train, x_val, y_val = load_probs(labels)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_train.shape)
    history = train(x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    main()
