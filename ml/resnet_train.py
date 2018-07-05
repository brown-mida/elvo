"""
This script allows you to train models by specifying a dictionary of
data source and hyperparameter values (in args).

To use this script, one would configure the arguments at the bottom
of the file and then run the script using nohup on a GPU machine.
You would then come back in a while to see your results (on Slack).

The code allows you to do the following things without much ML experience.
- select between data preprocessing techniques
- select between different models
- optimize hyperparameters
- evaluate models: choose a next step of action
  - tune parameters?
  - new model architecture?
  - better preprocessing/feature engineering?

The script assumes that:
- you already have processed data on your computer
- you are familiar with Python and the terminal
- you have ssh access to a cloud GPU

# TODO:
Better model evaluation:
- Feature importance
- PDF report generation (name, metrics, plots, params, etc.)
- Slack updates (upload the print statements/log)
- plot generation (ROC curve, PR curve, confusion matrix t-SNE, etc.)


TODO: from Michelangelo:
Who trained the model
Start and end time of the training job
Full model configuration (features used, hyper-parameter values, etc.)
Reference to training and test data sets
Distribution and relative importance of each feature
Model accuracy metrics
Standard charts and graphs for each model type (e.g. ROC curve, PR curve, and confusion matrix for a binary classifier)
Full learned parameters of the model
Summary statistics for model visualization
"""
import multiprocessing
import pathlib
import time
import typing

import keras
import numpy as np
import os
import pandas as pd
import sklearn.metrics
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


def split_data(x, y, split_idx):
    # TODO: split fraction instead of index
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_valid = x[split_idx:]
    y_valid = y[split_idx:]

    return x_train, y_train, x_valid, y_valid


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


def build_model(input_shape,
                dropout_rate1=0.5,
                dropout_rate2=0.5,
                optimizer=keras.optimizers.Adam(lr=1e-5),
                metrics=None) -> keras.models.Model:
    """Returns a compiled model ready for transfer learning.
    """
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


def create_generators(x_train, y_train, x_valid, y_valid, params):
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
    return train_gen, valid_gen


def create_callbacks(x_train, y_train, x_valid, y_valid):
    # TODO: Add back checkpointer later
    # checkpointer = keras.callbacks.ModelCheckpoint(
    #     filepath=params['model_path'],
    #     verbose=1,
    #     save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(patience=10)

    x_valid_standardized = (x_valid - x_train.mean) / x_train.std
    auc = AucCallback(x_valid_standardized, y_valid)
    return [early_stopper, auc]


class AucCallback(keras.callbacks.Callback):

    def __init__(self, x_valid_standardized, y_valid):
        super().__init__()
        self.x_valid_standardized = x_valid_standardized
        self.y_valid = y_valid

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_valid_standardized)
        score = sklearn.metrics.roc_auc_score(self.y_valid, y_pred)
        print(f'\nvalidation auc: {score}')


def create_model(x_train, y_train, x_valid, y_valid, params):
    """Fits a model"""
    train_gen, valid_gen = create_generators(x_train, y_train,
                                             x_valid, y_valid,
                                             params)
    model = build_model(input_shape=params['input_shape'],
                        dropout_rate1=params['dropout_rate1'],
                        dropout_rate2=params['dropout_rate2'])

    callbacks = create_callbacks(x_train, y_train, x_valid, y_valid)
    out = model.fit_generator(train_gen,
                              epochs=100,
                              validation_data=valid_gen,
                              verbose=2,
                              callbacks=callbacks)
    return out, model


def hyperoptimize(hyperparams):
    for i in range(len(hyperparams['data_dir']))[1:]:  # TODO: Remove [1:]
        print(f'using data {hyperparams["data_dir"][i]}')
        print(f'using labels {hyperparams["labels_path"][i]}')
        arrays = load_arrays(hyperparams['data_dir'][i])
        labels = pd.read_csv(hyperparams['labels_path'][i],
                             index_col=hyperparams['index_col'])[
            [hyperparams['label_col']]]

        print(f'seeding to {hyperparams["seed"]} before shuffling')
        np.random.seed(hyperparams["seed"])
        x, y = to_shuffled_arrays(arrays, labels)

        # TODO: Replace with more readable, generalizeable permutation code
        for batch_size in hyperparams['batch_size']:
            for dropout_rate1 in hyperparams['dropout_rate1']:
                for dropout_rate2 in hyperparams['dropout_rate2']:
                    for rotation_range in hyperparams['rotation_range']:
                        for split_idx in hyperparams['split_idx']:
                            params = {
                                'batch_size': batch_size,
                                'dropout_rate1': dropout_rate1,
                                'dropout_rate2': dropout_rate2,
                                'rotation_range': rotation_range,
                                'input_shape': hyperparams['input_shape'][i],
                            }
                            print(f'using params, {params}')
                            x_train, y_train, x_valid, y_valid = split_data(
                                x, y, split_idx)

                            print('training positives:', y_train.sum(),
                                  'training negatives',
                                  len(y_train) - y_train.sum())
                            print('validation positives:', y_valid.sum(),
                                  'validation negatives',
                                  len(y_valid) - y_valid.sum())
                            print('x_train mean:', x_train.mean(),
                                  'x_train std:', x_train.std())

                            # Run in a separate process to avoid memory
                            # issues
                            p = multiprocessing.Process(target=create_model,
                                                        args=(x_train, y_train,
                                                              x_valid, y_valid),
                                                        kwargs={
                                                            'params': params,
                                                        })
                            p.start()
                            p.join()


if __name__ == '__main__':
    # TODO: Consider a config file or command-line params for args
    args = {
        # Note: data_dir, labels_path, and input_shape must have the same
        # length.
        'data_dir': [
            '/home/lzhu7/elvo-analysis/data/processed-standard/arrays/',
            '/home/lzhu7/elvo-analysis/data/processed-no-basvert/arrays/',
            '/home/lzhu7/elvo-analysis/data/processed-220/labels.csv',
        ],
        'labels_path': [
            '/home/lzhu7/elvo-analysis/data/processed-standard/labels.csv',
            '/home/lzhu7/elvo-analysis/data/processed-no-basvert/labels.csv',
            '/home/lzhu7/elvo-analysis/data/processed-220/arrays/',
        ],
        'input_shape': [
            (200, 200, 3),
            (200, 200, 3),
            (220, 220, 3),
        ],
        'index_col': 'Anon ID',
        'label_col': 'occlusion_exists',
        'model_path': f'/home/lzhu7/elvo-analysis/models/'
                      f'model-{int(time.time())}.hdf5',
        ''
        'seed': 42,
        'split_idx': [700, 800, 900],
        'batch_size': [8, 16, 32, 64, 128],
        # TODO: Arguments for data augmentation parameters, etc.
        'dropout_rate1': [0.7, 0.8, 0.9],
        'dropout_rate2': [0.7, 0.8, 0.9],
        'rotation_range': [10, 20, 30],
    }

    hyperoptimize(args)
