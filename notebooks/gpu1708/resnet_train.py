import pathlib
import time
import typing

import functools
import keras
import numpy as np
import os
import pandas as pd
import sklearn.metrics
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


def load_arrays(data_dir: str) -> typing.Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def to_shuffled_arrays(data: typing.Dict[str, np.ndarray],
                       labels: pd.DataFrame) -> typing.Tuple[

    np.ndarray, np.ndarray]:
    shuffled_ids = list(data.keys())
    np.random.shuffle(shuffled_ids)
    X_list = []
    y_list = []
    for id_ in shuffled_ids:
        X_list.append(data[id_])
        y_list.append(labels.loc[id_])
    return np.stack(X_list), np.stack(y_list)


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def false_negatives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


class AucCallback(keras.callbacks.Callback):

    def __init__(self, x_valid_standardized, y_valid):
        super().__init__()
        self.x_valid_standardized = x_valid_standardized
        self.y_valid = y_valid

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_valid_standardized)
        score = sklearn.metrics.roc_auc_score(self.y_valid, y_pred)
        print(f'\nvalidation auc: {score}')


def build_model(input_shape,
                dropout_rate1=0.5,
                dropout_rate2=0.5,
                optimizer=keras.optimizers.Adam(lr=1e-5),
                metrics=None) -> keras.models.Model:
    resnet = keras.applications.ResNet50(include_top=False,
                                         input_shape=input_shape)
    x = resnet.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate1)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate2)(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(resnet.input, predictions)

    if metrics is None:
        print('Using default metrics: acc, sensitivity, specificity, tp, fn')
        metrics = ['acc',
                   sensitivity,
                   specificity,
                   true_positives,
                   false_negatives]

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model


def evaluate_generator(gen):
    print('generator mean:', gen.mean, 'generator std:', gen.std, )


def split_data(x, y, split_idx):
    print(f'splitting data to {split_idx} training samples',
          f' {len(x) - split_idx} validation samples')

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_valid = x[split_idx:]
    y_valid = y[split_idx:]

    print('training positives:', y_train.sum(),
          'training negatives', len(y_train) - y_train.sum())
    print('validation positives:', y_valid.sum(),
          'validation negatives', len(y_valid) - y_valid.sum())
    print('x_train mean:', x_train.mean(),
          'x_train std:', x_train.std())

    return x_train, y_train, x_valid, y_valid


def create_model(x_train, y_train, x_valid, y_valid, params):
    print('training positives:', y_train.sum(),
          'training negatives', len(y_train) - y_train.sum())
    print('validation positives:', y_valid.sum(),
          'validation negatives', len(y_valid) - y_valid.sum())
    print('x_train mean:', x_train.mean(),
          'x_train std:', x_train.std())

    train_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       rotation_range=params['rotation_range'],
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True)
    valid_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True)
    train_datagen.fit(x_train)
    valid_datagen.fit(x_train)

    train_gen = train_datagen.flow(x_train, y_train,
                                   batch_size=params['batch_size'])
    valid_gen = valid_datagen.flow(x_valid, y_valid,
                                   batch_size=params['batch_size'])

    evaluate_generator(train_datagen)
    evaluate_generator(valid_datagen)

    model = build_model(input_shape=params['input_shape'],
                        dropout_rate1=params['dropout_rate1'],
                        dropout_rate2=params['dropout_rate2'])

    # TODO: Add back checkpointer later
    # checkpointer = keras.callbacks.ModelCheckpoint(
    #     filepath=params['model_path'],
    #     verbose=1,
    #     save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(patience=10)

    x_valid_standardized = ((x_valid - valid_datagen.mean) /
                            valid_datagen.std)
    auc = AucCallback(x_valid_standardized, y_valid)

    out = model.fit_generator(train_gen,
                              epochs=100,
                              validation_data=valid_gen,
                              verbose=2,
                              callbacks=[auc, early_stopper])
    return out, model


if __name__ == '__main__':
    args = {
        'data_dir': [
            '/home/lzhu7/elvo-analysis/data/processed-standard/arrays/',
            '/home/lzhu7/elvo-analysis/data/processed-220/arrays/',
            '/home/lzhu7/elvo-analysis/data/processed-no-basvert/arrays/'
        ],
        'labels_path': [
            '/home/lzhu7/elvo-analysis/data/processed-standard/labels.csv',
            '/home/lzhu7/elvo-analysis/data/processed-220/labels.csv',
            '/home/lzhu7/elvo-analysis/data/processed-no-basvert/labels.csv',
        ],
        'index_col': 'Anon ID',
        'label_col': 'occlusion_exists',
        'model_path': f'/home/lzhu7/elvo-analysis/models/'
                      f'model-{int(time.time())}.hdf5',
        ''
        'seed': 42,
        'input_shape': [
            (200, 200, 3),
            (220, 220, 3),
            (200, 200, 3),
        ],
        'split_idx': [700, 800, 900],
        'batch_size': [8, 16, 32, 64, 128],
        # TODO: Arguments for data augmentation parameters, etc.
        'dropout_rate1': [0.5, 0.6, 0.7],
        'dropout_rate2': [0.5, 0.6, 0.7],
        'rotation_range': [10, 20, 30],
    }

    for i in range(len(args['data_dir'])):
        print(f'using data {args["data_dir"][i]}')
        print(f'using labels {args["labels_path"][i]}')
        arrays = load_arrays(args['data_dir'][i])
        labels = pd.read_csv(args['labels_path'][i],
                             index_col=args['index_col'])[[args['label_col']]]

        print(f'seeding to {args["seed"]}')
        np.random.seed(args["seed"])

        x, y = to_shuffled_arrays(arrays, labels)
        for batch_size in args['batch_size']:
            for dropout_rate1 in args['dropout_rate1']:
                for dropout_rate2 in args['dropout_rate2']:
                    for rotation_range in args['rotation_range']:
                        for split_idx in args['split_idx']:
                            params = {
                                'batch_size': batch_size,
                                'dropout_rate1': dropout_rate1,
                                'dropout_rate2': dropout_rate2,
                                'rotation_range': rotation_range,
                                'input_shape': args['input_shape'][i],
                            }
                            print(f'using params, {params}')
                            x_train, y_train, x_valid, y_valid = split_data(
                                x, y, split_idx)
                            create_model(x_train, y_train, x_valid, y_valid,
                                         params=params)
