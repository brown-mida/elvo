import os
import pathlib

import numpy as np
import pandas as pd
from keras import (
    applications,
    layers,
    models,
    optimizers,
    callbacks,
)
from keras import backend as K
from keras.preprocessing import image


def load_data(data_dir):
    """Returns a dictionary which maps patient ids
    to patient pixel data."""
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def to_arrays(data, labels):
    shuffled_ids = list(data.keys())
    np.random.shuffle(shuffled_ids)
    X_list = []
    y_list = []
    for id_ in shuffled_ids:
        X_list.append(data[id_])
        y_list.append(labels.loc[id_])
    return np.stack(X_list), np.stack(y_list)


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def build_model():
    resnet = applications.ResNet50(include_top=False,
                                   input_shape=(200, 200, 3))

    x = resnet.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(resnet.input, predictions)

    #     for layer in model.layers[:10]:
    #         layer.trainable = False

    model.compile(optimizer=optimizers.Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['acc', sensitivity, specificity])

    return model


if __name__ == '__main__':
    data = load_data('/home/lzhu7/elvo-analysis/data/mip_three')
    labels = pd.read_csv('/home/lzhu7/elvo-analysis/data/labels_mip_three.csv',
                         index_col='patient_id')
    x, y = to_arrays(data, labels)

    x_train = x[0:800]
    y_train = y[0:800]
    x_valid = x[800:]
    y_valid = y[800:]
    print('training positives:', y_train.sum())
    print('validation positives:', y_valid.sum())

    datagen = image.ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       rotation_range=30,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       horizontal_flip=True)
    datagen.fit(x_train)
    train_gen = datagen.flow(x_train, y_train)
    valid_gen = datagen.flow(x_valid, y_valid)

    model = build_model()
    checkpointer = callbacks.ModelCheckpoint(filepath='weights2.hdf5',
                                             verbose=1, save_best_only=True)
    early_stopper = callbacks.EarlyStopping(patience=5)
    model.fit_generator(train_gen, epochs=100, validation_data=valid_gen,
                        callbacks=[checkpointer, early_stopper])
