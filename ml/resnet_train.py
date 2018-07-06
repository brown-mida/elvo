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
Standard charts and graphs for each model type
(e.g. ROC curve, PR curve, and confusion matrix for a binary classifier)
Full learned parameters of the model
Summary statistics for model visualization
"""
import multiprocessing
import pathlib
import time
import typing
from pprint import pprint

import keras
import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection

import utils
from models import luke


def load_arrays(data_dir: str) -> typing.Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def to_arrays(data: typing.Dict[str, np.ndarray],
              labels: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Converts the data and labels into numpy arrays.

    Note: This function filters mismatched labels.

    Note: The index of labels must be patient IDs.

    :param data:
    :param labels: a dataframe WITH patient ID for the index.
    :return: two arrays containing the arrays and the labels
    """
    patient_ids = data.keys()
    X_list = []
    y_list = []
    for id_ in patient_ids:
        try:
            y_list += [labels.loc[id_]]
            X_list += [data[id_]]  # Needs to be in this order
        except KeyError:
            print(f'{id_} in data was not present in labels')
            print(len(X_list), len(y_list))
    for id_ in labels.index.values:
        if id_ not in patient_ids:
            print(f'{id_} in labels was not present in data')
    return np.stack(X_list), np.stack(y_list)


def create_generators(x_train, y_train, x_valid, y_valid,
                      rotation_range,
                      batch_size):
    train_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       rotation_range=rotation_range,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True)
    valid_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True)
    train_datagen.fit(x_train)
    valid_datagen.fit(x_train)

    train_gen = train_datagen.flow(x_train, y_train,
                                   batch_size=batch_size)
    valid_gen = valid_datagen.flow(x_valid, y_valid,
                                   batch_size=batch_size)
    return train_gen, valid_gen


def create_callbacks(x_train: np.ndarray,
                     y_train: np.ndarray,
                     x_valid: np.ndarray,
                     y_valid: np.ndarray):
    # TODO: Add back checkpointer later
    # checkpointer = keras.callbacks.ModelCheckpoint(
    #     filepath=params['model_path'],
    #     verbose=1,
    #     save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(patience=10)

    x_mean = np.array([x_train[:, :, :, 0].mean(),
                       x_train[:, :, :, 1].mean(),
                       x_train[:, :, :, 2].mean()])
    x_std = np.array([x_train[:, :, :, 0].std(),
                      x_train[:, :, :, 1].std(),
                      x_train[:, :, :, 2].std()])
    x_valid_standardized = (x_valid - x_mean) / x_std

    auc = utils.AucCallback(x_valid_standardized, y_valid)
    return [early_stopper, auc]


def create_job(x_train, y_train, x_valid, y_valid, model_params):
    """Builds, fits, and evaluates a model"""
    # TODO: Not all future models will use this generator
    train_gen, valid_gen = create_generators(x_train, y_train,
                                             x_valid, y_valid,
                                             model_params['rotation_range'],
                                             model_params['batch_size'])

    # TODO: Not all models will have the following parameters
    model = model_params['model_callable'](
        input_shape=x_train.shape[1:],
        dropout_rate1=model_params['dropout_rate1'],
        dropout_rate2=model_params['dropout_rate2']
    )

    print('Using default metrics: acc, sensitivity, specificity, tp, fn')
    metrics = ['acc',
               utils.sensitivity,
               utils.specificity,
               utils.true_positives,
               utils.false_negatives]

    model.compile(optimizer=model_params['optimizer'],
                  loss='binary_crossentropy',
                  metrics=metrics)

    callbacks = create_callbacks(x_train, y_train, x_valid, y_valid)
    out = model.fit_generator(train_gen,
                              epochs=100,
                              validation_data=valid_gen,
                              verbose=2,
                              callbacks=callbacks)

    return out, model


def hyperoptimize(hyperparams):
    param_grid = model_selection.ParameterSampler(hyperparams, n_iter=10)
    for params in param_grid:
        pprint(f'params:\n{params}', indent=4)
        data_params = params['data']
        array_dict = load_arrays(data_params['data_dir'])
        index_col = data_params['index_col']
        label_col = data_params['label_col']
        label_df = pd.read_csv(data_params['labels_path'],
                               index_col=index_col)[[label_col]]

        x, y = to_arrays(array_dict, label_df)
        print(f'seeding to {params["seed"]} before shuffling')

        x_train, x_valid, y_train, y_valid = \
            model_selection.train_test_split(
                x, y,
                test_size=params['val_split'],
                random_state=params["seed"])

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
        # TODO: Use tensorflow to utilize more GPU compute
        p = multiprocessing.Process(target=create_job,
                                    args=(x_train, y_train,
                                          x_valid, y_valid),
                                    kwargs={
                                        'model_params': params['model'],
                                    })
        p.start()
        p.join()
        time.sleep(10)  # Sleep to avoid memory errors


if __name__ == '__main__':
    # TODO: Consider a config file or command-line params for args
    # TODO: Consider turning the dict into a class for better attribute clarity

    # All top-level values should be lists
    arguments = {
        'data': [
            {
                # A directory containing a list of numpy files with
                # patient ID as their filename
                'data_dir': '/home/lzhu7/elvo-analysis/data/'
                            'processed-standard/arrays/',
                # A CSV file generated by saving a pandas DataFrame
                'labels_path': '/home/lzhu7/elvo-analysis/data/'
                               'processed-standard/labels.csv',
                'index_col': 'Anon ID',
                'label_col': 'occlusion_exists',
            },
            {
                'data_dir': '/home/lzhu7/elvo-analysis/data/'
                            'processed-no-basvert/arrays/',
                'labels_path': '/home/lzhu7/elvo-analysis/data/'
                               'processed-no-basvert/labels.csv',
                'index_col': 'Anon ID',
                'label_col': 'occlusion_exists',
            },
            {
                'data_dir': '/home/lzhu7/elvo-analysis/data/'
                            'mip_transform/',
                # TODO: This is not ideal
                'labels_path': '/home/lzhu7/elvo-analysis/data/'
                               'processed-standard/labels.csv',
                'index_col': 'Anon ID',
                'label_col': 'occlusion_exists',
            },
            {
                'data_dir': '/home/lzhu7/elvo-analysis/data/'
                            'processed-lower/arrays/',
                'labels_path': '/home/lzhu7/elvo-analysis/data/'
                               'processed-lower/labels.csv',
                'index_col': 'Anon ID',
                'label_col': 'occlusion_exists',
            }
        ],

        # 'model_save_path': f'/home/lzhu7/elvo-analysis/models/'
        #                    f'model-{int(time.time())}.hdf5',

        'seed': [0, 42],
        'val_split': [0.2, 0.3],

        # TODO: Reduce complexity
        'model': [{
            'model_callable': luke.resnet,
            'dropout_rate1': 0.8,
            'dropout_rate2': 0.8,
            'batch_size': 8,
            'rotation_range': 20,
            'optimizer': keras.optimizers.Adam(lr=1e-5)

        }, {
            'model_callable': luke.resnet,
            'dropout_rate1': 0.8,
            'dropout_rate2': 0.8,
            'batch_size': 8,
            'rotation_range': 20,
            'optimizer': keras.optimizers.Adam(lr=1e-5)

        }, {
            'model_callable': luke.resnet,
            'dropout_rate1': 0.8,
            'dropout_rate2': 0.8,
            'batch_size': 8,
            'rotation_range': 20,
            'optimizer': keras.optimizers.Adam(lr=1e-5)
        }],
    }

    hyperoptimize(arguments)
