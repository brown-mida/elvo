import tensorflow as tf
from models.three_d import cube_classifier
from keras.optimizers import Adadelta, SGD
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
import pandas as pd
import numpy as np
from google.cloud import storage
import io
import os

BLACKLIST = []
LEARN_RATES = [1e-4, 5e-5]

DROPOUTS = [(0.4, 0.65),
            (0.45, 0.7),
            (0.5, 0.75),
            (0.4, 0.4),
            (0.5, 0.5)]


def download_array(blob: storage.Blob) -> np.ndarray:
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def find_best_models(results):
    max_accs = {}
    avg_accs = {}
    var_accs = {}
    for model, result in list(results.items()):
        max_accs[model] = result['max']
        avg_accs[model] = result['avg']
        var_accs[model] = result['var']

    sorted_max = [(model, max_accs[model]) for model in sorted(max_accs, key=max_accs.get, reverse=True)]
    sorted_avg = [(model, avg_accs[model]) for model in sorted(avg_accs, key=avg_accs.get, reverse=True)]
    sorted_var = [(model, var_accs[model]) for model in sorted(var_accs, key=var_accs.get, reverse=True)]
    print('\n\n------------------------\nMODELS RANKED BY MAX ACC\n------------------------\n')
    for i, result in enumerate(sorted_max):
        print(i, result)

    print('\n\n------------------------\nMODELS RANKED BY AVG ACC\n------------------------\n')
    for i, result in enumerate(sorted_avg):
        print(i, result)

    print('\n\n------------------------\nMODELS RANKED BY VAR ACC\n------------------------\n')
    for i, result in enumerate(sorted_var):
        print(i, result)


def train(x_train, y_train, x_val, y_val):
    models = {}
    results = []
    for lr in LEARN_RATES:
        for dropout in DROPOUTS:
            print(f'\n\n---------------------------------------------------\n'
                  f'TRAINING MODEL : LR = {lr}, DROPOUT = {dropout}'
                  f'\n---------------------------------------------------\n')

            for i in range(10):
                model = cube_classifier.CubeClassifierBuilder.build(dropout=dropout)
                model.compile(loss=categorical_crossentropy,
                              optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
                              metrics=['accuracy'])
                model.fit(x=x_train,
                          y=y_train,
                          batch_size=128,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)],
                          epochs=300,
                          validation_data=(x_val, y_val))

                result = model.evaluate(x_val, y_val, verbose=1)
                results = np.append(results, result[1])
            results = np.asarray(results)
            models[f'lr: {lr}, dropout: {dropout}'] = {'max': np.max(results),
                                                       'avg': np.mean(results),
                                                       'var': np.var(results)}

    return models


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
    for idx, blob in enumerate(bucket.list_blobs(prefix='chunk_data/preds/train')):
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

    x_val = []
    y_val = []
    for idx, blob in enumerate(bucket.list_blobs(prefix='chunk_data/preds/val')):
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
    results = train(x_train, y_train, x_val, y_val)
    find_best_models(results)


if __name__ == '__main__':
    main()
