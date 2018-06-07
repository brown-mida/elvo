"""Trains a model on a department machine.

Make sure to copy the data from thingumy to here first.
"""
import logging
import os
import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers

# This allows us to import from models and generators
root_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(root_dir)

LENGTH, WIDTH, HEIGHT = (150, 150, 64)  # TODO


def load_training_data() -> np.array:
    """Returns a 4D matrix of the training data.

     The data is in the form (n_samples, l, w, h). The samples
     are sorted by patient ID.
     """
    arrays = []
    training_filenames = sorted(os.listdir(
        '/home/lzhu7/data/numpy_split/training'))  # TODO: Remove limit
    for filename in training_filenames:
        arr = np.load('/home/lzhu7/data/numpy_split/training/' + filename)
        if arr.shape == (LENGTH, WIDTH, HEIGHT):
            arrays.append(arr)
        else:
            logging.info(
                f'training file {filename} has incorrect shape {arr.shape}')
    return np.stack(arrays)


def load_validation_data() -> np.array:
    """Returns a 4D matrix of the validation data.

     The data is in the form (n_samples, l, w, h). The samples
     are sorted by patient ID.
    """
    arrays = []
    validation_filenames = sorted(os.listdir(
        '/home/lzhu7/data/numpy_split/validation'))
    for filename in validation_filenames:
        arr = np.load('/home/lzhu7/data/numpy_split/validation/' + filename)
        if arr.shape == (LENGTH, WIDTH, HEIGHT):
            arrays.append(arr)
        else:
            logging.info(
                f'validation file {filename} has incorrect shape {arr.shape}')
    return np.stack(arrays)


def load_labels() -> (np.array, np.array):
    training_df = pd.read_csv('/home/lzhu7/data/training_labels.csv')
    validation_df = pd.read_csv('/home/lzhu7/data/validation_labels.csv')
    training_labels = training_df.sort_values('patient_id')['label'].values
    validation_labels = validation_df.sort_values('patient_id')['label'].values
    return training_labels, validation_labels


def build_model() -> keras.Model:
    """Returns a compiled model.
    """
    model = keras.Sequential()
    model.add(layers.Conv2D(256,
                            (3, 3),
                            activation='relu',
                            input_shape=(LENGTH, WIDTH, HEIGHT)))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Dense(1024))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


if __name__ == '__main__':
    X_train = load_training_data()
    logging.info(f'loaded training data with shape {X_train.shape}')
    X_valid = load_validation_data()
    logging.info(f'loaded validation data with shape {X_valid.shape}')
    y_train, y_valid = load_labels()
    logging.info(f'loaded training labels with shape {y_train.shape}')
    logging.info(f'loaded validation labels with shape {y_valid.shape}')

    model = build_model()
    print(model.summary())
    model.fit(X_train, y_train,
              batch_size=32, epochs=10, validation_data=(X_valid, y_valid))
