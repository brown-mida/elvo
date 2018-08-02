"""
This script serves serves as a way to train the C3D model on subsets of the
training data of varying sizes, reporting the results of that training to
the model_results channel on Slack.
"""
import datetime
import logging
import subprocess

import keras
import numpy as np

from . import plotting
from . import utils
from .model import C3DBuilder


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
                y_valid: np.ndarray):
    """
    Train a C3D model on the given data.

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :return:
    """
    # To avoid data type issues in the Keras library calls
    if x_train.shape[1:] != (32, 32, 32) or x_valid.shape[1:] != (32, 32, 32):
        raise ValueError(
            'x_train and x_valid should have shape (?, 32, 32,32), got {} '
            'and {}'.format(
                x_train.shape, x_valid.shape
            ))

    if y_train.ndim != 1 or y_valid.ndim != 1:
        raise ValueError(
            'y_train and y_valid should have shape (?,), got {} and {}'.format(
                y_train.shape, y_valid.shape
            ))

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int16)
    x_valid = x_valid.astype(np.float32)
    y_valid = y_valid.astype(np.int16)

    x_train = np.expand_dims(x_train, axis=-1)
    x_valid = np.expand_dims(x_valid, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_valid = np.expand_dims(y_valid, axis=-1)

    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std

    describe_data(x_train, y_train, x_valid, y_valid)

    metrics = ['acc',
               utils.true_positives,
               utils.false_negatives,
               utils.sensitivity,
               utils.specificity]
    model = C3DBuilder.build()
    opt = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt,
                  loss={"out_class": "binary_crossentropy"},
                  metrics=metrics)

    callbacks = utils.create_callbacks(x_train=x_train,
                                       y_train=y_train,
                                       x_valid=x_valid,
                                       y_valid=y_valid,
                                       normalize=False)
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=50,
                        batch_size=8,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks,
                        verbose=2)

    now = datetime.datetime.utcnow().isoformat()
    plotting.upload_gcs_plots(x_train,
                              x_valid,
                              y_valid,
                              model,
                              history,
                              job_name='c3d-luke',
                              created_at=now,
                              chunk=True)


def describe_data(x_train: np.ndarray, y_train: np.ndarray,
                  x_valid: np.ndarray, y_valid: np.ndarray):
    logging.info(
        'training data: {} samples, {} positives'.format(y_train.shape[0],
                                                         y_train.sum()))
    logging.info(
        'validation data: {} samples, {} positives'.format(y_valid.shape[0],
                                                           y_valid.sum()))
    logging.info(
        'x_train: mean {}, std {}'.format(x_train.mean(), x_train.std()))
    logging.info(
        'x_valid: mean {}, std {}'.format(x_valid.mean(), x_valid.std()))


def run(data_name: str):
    subprocess.call(['gsutil',
                     'cp',
                     'gs://elvos/processed3d/{data_name}/x_train.npy'.format(
                         data_name=data_name),
                     '/tmp/'])
    subprocess.call(['gsutil',
                     'cp',
                     'gs://elvos/processed3d/{data_name}/y_train.npy'.format(
                         data_name=data_name),
                     '/tmp/'])
    subprocess.call(['gsutil',
                     'cp',
                     'gs://elvos/processed3d/{data_name}/x_valid.npy'.format(
                         data_name=data_name),
                     '/tmp/'])
    subprocess.call(['gsutil',
                     'cp',
                     'gs://elvos/processed3d/{data_name}/y_valid.npy'.format(
                         data_name=data_name),
                     '/tmp/'])
    x_train = np.load('/tmp/x_train.npy')
    y_train = np.load('/tmp/y_train.npy')
    x_valid = np.load('/tmp/x_valid.npy')
    y_valid = np.load('/tmp/y_valid.npy')
    train_model(x_train, y_train, x_valid, y_valid)


if __name__ == '__main__':
    fmt = '[%(asctime)s] {%(filename)s} %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG)
    run('airflow-2')

# TODO(luke): Add layer to c3d for binary classification
