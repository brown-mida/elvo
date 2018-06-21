"""3D convolutional autoencoder.

Potential use-case: dimensionality reduction for t-SNE.
"""
import os
import pathlib
import random
import typing

import numpy as np
import pandas as pd
from keras import models, layers


def load_data(data_dir: str) -> typing.Dict[str, np.ndarray]:
    """Returns a dictionary which maps patient ids
    to patient pixel data."""
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(
            pathlib.Path(data_dir) / filename)
    return data_dict


def prepare_for_training(data: typing.Dict[str, np.ndarray],
                         labels: pd.DataFrame) -> (np.ndarray, np.ndarray):
    shuffled_ids = list(data.keys())
    random.shuffle(shuffled_ids)
    X_list = []
    y_list = []
    for id_ in shuffled_ids:
        X_list.append(data[id_])
        y_list.append(labels.loc[id_])
    return np.stack(X_list), np.stack(y_list)


def create_autoencoder():
    model = models.Sequential()
    model.add(layers.Conv3D(64, 5, padding='same', activation='relu',
                            input_shape=(80, 80, 64, 1)))
    model.add(layers.MaxPool3D())
    model.add(layers.Conv3D(128, 5, padding='same', activation='relu'))
    model.add(layers.MaxPool3D())
    model.add(layers.Conv3D(128, 5, padding='same', activation='relu'))
    model.add(layers.MaxPool3D())
    model.add(layers.Conv3D(256, 5, padding='same', activation='relu'))
    model.add(layers.Deconv3D(256, 4, padding='same', activation='relu'))
    model.add(layers.UpSampling3D())
    model.add(layers.Deconv3D(128, 4, padding='same', activation='relu'))
    model.add(layers.UpSampling3D())
    model.add(layers.Deconv3D(128, 4, padding='same', activation='relu'))
    model.add(layers.UpSampling3D())
    model.add(layers.Deconv3D(64, 4, padding='same', activation='relu'))
    model.add(layers.Deconv3D(1, 4, padding='same', activation='relu'))
    return model


if __name__ == '__main__':
    data_path = '/home/lzhu7/elvo-analysis/data/luke4/'
    processed_dict = load_data(data_path)

    labels_path = '/home/lzhu7/elvo-analysis/data/labels4.csv'
    labels_df = pd.read_csv(labels_path, index_col='patient_id')

    X, y = prepare_for_training(processed_dict, labels_df)
    X = np.expand_dims(X, axis=4)
    print('X has shape:', X.shape)

    model = create_autoencoder()
    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy')
    model.summary()

    for i in range(50):
        model.fit(X, X, epochs=1, batch_size=8, validation_split=0.2,
                  verbose=2)
        if i % 5 == 0:
            print('saving model')
            model.save(f'convolutional_autoencoder_3d-{i}.hdf5')
