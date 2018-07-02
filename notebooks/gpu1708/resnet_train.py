import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
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


def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def build_resnet_model():
    resnet = applications.ResNet50(include_top=False, input_shape=(220, 220, 3))
    x = resnet.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    
#     for layer in resnet.layers:
#         layer.trainable = False

    model = models.Model(resnet.input, predictions)
    
    auc_roc = as_keras_metric(tf.metrics.auc)
    true_positives = as_keras_metric(tf.metrics.true_positives)
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model.compile(optimizer=optimizers.Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['acc', auc_roc, precision, recall, true_positives])

    return model


if __name__ == '__main__':
    data = load_data('/home/lzhu7/elvo-analysis/data/mip_three16/')
    labels = pd.read_csv('/home/lzhu7/elvo-analysis/data/labels_mip_three16.csv',
                         index_col='Anon ID')[['occlusion_exists']]
    print('seeding to 0')
    np.random.seed(42)
    x, y = to_arrays(data, labels)

    x_train = x[0:900]
    y_train = y[0:900]
    x_valid = x[900:]
    y_valid = y[900:]
    print('training positives:', y_train.sum(), 'training negatives', len(y_train) - y_train.sum())
    print('validation positives:', y_valid.sum(),  'validation negatives', len(y_valid) - y_valid.sum())
    print('x_train mean:', x_train.mean(), 'x_train std:', x_train.std())

    datagen = image.ImageDataGenerator(rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=[1.0, 1.1],
                                       horizontal_flip=True)
    datagen.fit(x_train)
    train_gen = datagen.flow(x_train, y_train, batch_size=48)
    valid_gen = datagen.flow(x_valid, y_valid, batch_size=48)
    
    train_arr = train_gen.next()[0]
    valid_arr = valid_gen.next()[0]
    print('shape:', train_arr.shape, 'mean:', train_arr.mean(), 'std:', train_arr.std())
    print('shape:', valid_arr.shape, 'mean:', valid_arr.mean(), 'std:', valid_arr.std())

    model = build_resnet_model()
    model.summary()
    checkpointer = callbacks.ModelCheckpoint(filepath='weights-13.hdf5',
                                             verbose=1, save_best_only=True)
    early_stopper = callbacks.EarlyStopping(patience=10)
    model.fit_generator(train_gen, epochs=100, validation_data=valid_gen,
                        callbacks=[checkpointer, early_stopper])
