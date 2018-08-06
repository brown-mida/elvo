# TODO: get this to work — try it with second-stage trained models
"""
This script attempts to find the best hyperparameters for training the
classifier model with, and just generally trains the classifer model on
probability distribution data.
"""

import tensorflow as tf
from ml.models.three_d import cube_classifier
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
import pandas as pd
import numpy as np
from google.cloud import storage
from scipy.ndimage.interpolation import zoom
import io
import os

BLACKLIST = []
# Initially, this included values from 1e-3 to 1e-7; these two performed best
LEARN_RATES = [1e-4, 5e-5]

DROPOUTS = [(0.4, 0.65),
            (0.45, 0.7),
            (0.5, 0.75),
            (0.4, 0.4),
            (0.5, 0.5)]


def download_array(blob: storage.Blob) -> np.ndarray:
    """Downloads a blob to a numpy array

    :param blob: GCS blob to download as a numpy array
    :return: numpy array
    """
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def find_best_models(results):
    """Sorts/prints out the best performing models by max accuracy, average
     accuracy, and variance in accuracies across runs

    :param results: dict of avg acc, max acc, and variance in accuracy
    :return:
    """
    max_accs = {}
    avg_accs = {}
    var_accs = {}

    # Copy results to individual dicts
    for model, result in list(results.items()):
        max_accs[model] = result['max']
        avg_accs[model] = result['avg']
        var_accs[model] = result['var']

    # Sort results by highest performance
    sorted_max = [(model, max_accs[model])
                  for model in sorted(max_accs,
                                      key=max_accs.get,
                                      reverse=True)]
    sorted_avg = [(model, avg_accs[model])
                  for model in sorted(avg_accs,
                                      key=avg_accs.get,
                                      reverse=True)]
    sorted_var = [(model, var_accs[model])
                  for model in sorted(var_accs,
                                      key=var_accs.get)]

    # Print max accuracy results
    print('\n\n------------------------\n'
          'MODELS RANKED BY MAX ACC'
          '\n------------------------\n')
    for i, result in enumerate(sorted_max):
        print(i, result)

    # Print avg accuracy results
    print('\n\n------------------------\n'
          'MODELS RANKED BY AVG ACC'
          '\n------------------------\n')
    for i, result in enumerate(sorted_avg):
        print(i, result)

    # Print accuracy variance results
    print('\n\n------------------------\n'
          'MODELS RANKED BY VAR ACC'
          '\n------------------------\n')
    for i, result in enumerate(sorted_var):
        print(i, result)


def train(x_train, y_train, x_val, y_val, x_test, y_test):
    """A function that iteratively trains the model on a grid search of
    parameters for learning rate and dropout

    :param x_train: training data
    :param y_train: training labels
    :param x_val: validation data
    :param y_val: validation labels
    :param x_test: test data
    :param y_test: test labels
    :return:
    """
    models = {}
    results = []
    # iterate through potential learning rates and dropouts
    for lr in LEARN_RATES:
        for dropout in DROPOUTS:
            print(f'\n\n---------------------------------------------------\n'
                  f'TRAINING MODEL : LR = {lr}, DROPOUT = {dropout}'
                  f'\n---------------------------------------------------\n')

            # do 10 iterations per model
            for i in range(10):
                # make the model
                model = cube_classifier.\
                    CubeClassifierBuilder.build(dropout=dropout, binary=True)
                opt = SGD(lr=1e-5, momentum=0.9, nesterov=True)
                # compite the model
                model.compile(loss=binary_crossentropy,
                              optimizer=opt,
                              metrics=['accuracy'])
                # fit to x_train and y_train, stopping based off of val set
                model.fit(x=x_train,
                          y=y_train,
                          batch_size=128,
                          callbacks=[EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   verbose=1)],
                          epochs=300,
                          validation_data=(x_val, y_val))
                # evaluate based on test set
                result = model.evaluate(x_test, y_test)
                results = np.append(results, result[1])
            results = np.asarray(results)
            # calculate max, mean and variance for each config of the model
            models[f'lr: {lr}, dropout: {dropout}'] = {'max': np.max(results),
                                                       'avg': np.mean(results),
                                                       'var': np.var(results)}
    return models


def load_probs(labels: pd.DataFrame):
    """
    Loads a set of probabilities that any chunk is an ELVO (to be used in our
    binary/categorical classifier based off 3D data)
    :param labels: the set of labels from load_labels()
    :return: training data/labels, validation data/labels, test data/labels
    """
    print(labels)
    # Get pred files from Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        # 'credentials/client_secret.json'
    )
    labels.set_index('ID', inplace=True)
    bucket = gcs_client.get_bucket('elvos')

    x_train = []
    y_train = []
    # Get training preds
    blobs = bucket.list_blobs(prefix='chunk_data/preds/train')
    for idx, blob in enumerate(blobs):
        if idx % 10 == 0:
            print(f'successfully loaded {idx} training scans and labels')
        file_id = blob.name.split('/')[-1].split('.')[0]
        arr = download_array(blob)
        # If this ID is in the set of ID's we know
        if file_id in labels.index.values:
            # Scale it to a list of 784 values and reshape to a 28x28 square
            arr = zoom(arr, (784 / len(arr)))
            arr = np.asarray(arr)
            arr = np.reshape(arr, (28, 28, 1))
            # Append to x_train
            x_train.append(arr)

            # Append the corresponding label to y_train
            label = np.array([labels.loc[file_id]['Label']])
            label = label.astype(int)
            y_train.append(label)

    x_val = []
    y_val = []
    # Get val preds
    blobs = bucket.list_blobs(prefix='chunk_data/preds/val')
    for idx, blob in enumerate(blobs):
        if idx % 10 == 0:
            print(f'successfully loaded {idx} validation scans and labels')
        file_id = blob.name.split('/')[-1].split('.')[0]
        arr = download_array(blob)
        # If this ID is in the set of ID's we know
        if file_id in labels.index.values:
            # Scale it to a list of 784 values and reshape to a 28x28 square
            arr = zoom(arr, (784 / len(arr)))
            arr = np.asarray(arr)
            arr = np.reshape(arr, (28, 28, 1))
            # Append to x_val
            x_val.append(arr)

            # Append the corresponding label to y_val
            label = np.array([labels.loc[file_id]['Label']])
            label = label.astype(int)
            y_val.append(label)

    x_test = []
    y_test = []
    # Get test preds
    blobs = bucket.list_blobs(prefix='chunk_data/preds/test')
    for idx, blob in enumerate(blobs):
        if idx % 10 == 0:
            print(f'successfully loaded {idx} test scans and labels')
        file_id = blob.name.split('/')[-1].split('.')[0]
        arr = download_array(blob)
        # If this ID is in the set of ID's we know
        if file_id in labels.index.values:
            # Scale it to a list of 784 values and reshape to a 28x28 square
            arr = zoom(arr, (784 / len(arr)))
            arr = np.asarray(arr)
            arr = np.reshape(arr, (28, 28, 1))
            # Append to x_test
            x_test.append(arr)

            # Append the corresponding label to y_test
            label = np.array([labels.loc[file_id]['Label']])
            label = label.astype(int)
            y_test.append(label)

    # return train/val/test arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_labels():
    """
    Loads binary labels
    :return:
    """
    return pd.read_csv('labels.csv')


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.Session(config=tf.ConfigProto(log_device_placement=True))

    labels = load_labels()
    x_train, y_train, x_val, y_val, x_test, y_test = load_probs(labels)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)
    results = train(x_train, y_train, x_val, y_val, x_test, y_test)
    find_best_models(results)


if __name__ == '__main__':
    main()
