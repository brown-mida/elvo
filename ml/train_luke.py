"""Trains a model on a department machine.

Make sure to copy the data from thingumy to here first.
"""
import logging
import os
import time

import keras
import numpy as np
import pandas as pd
from keras import layers, optimizers

LENGTH, WIDTH, HEIGHT = (120, 120, 64)

VALID_TRAINING_INDICES = []
VALID_VALIDATION_INDICES = []


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def load_training_data() -> np.ndarray:
    """Returns a 4D matrix of the training data.

     The data is in the form (n_samples, l, w, h). The samples
     are sorted by patient ID.
     """
    arrays = []
    training_filenames = sorted(os.listdir(
        '/home/lzhu7/data/numpy_split/training'))
    for i, filename in enumerate(training_filenames):
        arr = np.load('/home/lzhu7/data/numpy_split/training/' + filename)
        if arr.shape == (LENGTH, WIDTH, HEIGHT):
            arrays.append(arr)
            VALID_TRAINING_INDICES.append(i)
        else:
            logging.info(
                f'training file {filename} has incorrect shape {arr.shape}')
    return np.stack(arrays)


def load_validation_data() -> np.ndarray:
    """Returns a 4D matrix of the validation data.

     The data is in the form (n_samples, l, w, h). The samples
     are sorted by patient ID.
    """
    arrays = []
    validation_filenames = sorted(os.listdir(
        '/home/lzhu7/data/numpy_split/validation'))
    for i, filename in enumerate(validation_filenames):
        arr = np.load('/home/lzhu7/data/numpy_split/validation/' + filename)
        if arr.shape == (LENGTH, WIDTH, HEIGHT):
            arrays.append(arr)
            VALID_VALIDATION_INDICES.append(i)
        else:
            logging.info(
                f'validation file {filename} has incorrect shape {arr.shape}')
    return np.stack(arrays)


def load_labels() -> (np.ndarray, np.ndarray):
    training_df = pd.read_csv('/home/lzhu7/data/training_labels.csv')
    validation_df = pd.read_csv('/home/lzhu7/data/validation_labels.csv')
    training_labels = training_df.sort_values('patient_id')['label'].values
    validation_labels = validation_df.sort_values('patient_id')['label'].values
    return training_labels, validation_labels


def normalize(X: np.ndarray, mean, std):
    return (X - mean) / std


def squash_height(X: np.ndarray):
    assert X.ndim == 4
    return X.max(axis=3)


def build_model() -> keras.Model:
    """Returns a compiled model.

    This model takes in an array of (120, 120, 64) images and returns 0 or 1.
    """
    model = keras.Sequential()
    model.add(layers.Conv2D(32,
                            (3, 3),
                            activation='relu',
                            input_shape=(LENGTH, WIDTH, HEIGHT, 1),
                            use_bias=False))
    model.add(layers.Conv2D(32, (3, 3),
                            activation='relu',
                            use_bias=False,
                            padding='same'))
    model.add(layers.MaxPool3D())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', use_bias=False,
                            padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=False,
                            padding='same'))
    model.add(layers.MaxPool3D())
    model.add(layers.Conv2D(1024, (3, 3), activation='relu', use_bias=False,
                            padding='same'))
    model.add(layers.Conv2D(1024, (3, 3), activation='relu', use_bias=False,
                            padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu', use_bias=False))
    #     model.add(layers.Dense(1024, activation='relu', use_bias=False))
    #     model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid', use_bias=False))

    model.compile(optimizer=optimizers.Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    X_train = load_training_data()
    logging.info(f'loaded training data with shape {X_train.shape}')
    y_train, _ = load_labels()
    logging.info(f'loaded training labels with shape {y_train.shape}')
    y_train = y_train[VALID_TRAINING_INDICES]
    logging.info(f'filtered training labels to shape {y_train.shape}')

    X_valid = load_validation_data()
    logging.info(f'loaded validation data with shape {X_valid.shape}')
    _, y_valid = load_labels()
    logging.info(f'loaded validation labels with shape {y_valid.shape}')
    y_valid = y_valid[VALID_VALIDATION_INDICES]
    logging.info(f'filtered validation labels to shape {y_valid.shape}')

    logging.info('normalizing data')
    X_mean = X_train.mean()
    X_std = X_train.std()
    X_train = normalize(X_train, X_mean, X_std)
    X_valid = normalize(X_valid, X_mean, X_std)

    logging.info('building the model')
    model = build_model()
    model.summary()
    # Overfit -> tune
    model.fit(X_train[0:500], y_train[0:500],
              batch_size=32,
              epochs=5,
              validation_data=(X_valid, y_valid))

    logging.info(
        f'predictions on X_train[0:10]: {model.predict(X_train[0:10])}')
    logging.info(
        f'predictions on X_valid[0:10]: {model.predict(X_valid[0:10])}')

    inp = input('Enter "y" to save the model:')
    if inp == 'y':
        filename = f'model-{time.time()}.hdf5'
        logging.info(f'Saving model to {filename}')
        model.save(filename)


if __name__ == '__main__':
    configure_logger()
    main()
