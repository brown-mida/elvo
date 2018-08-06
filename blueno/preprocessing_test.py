import os

import keras
import numpy as np
import pandas as pd
import pytest

import blueno
from blueno import preprocessing


def small_model(input_shape=(224, 224, 3),
                num_classes=1,
                *args, **kwargs):
    return keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])


def test_to_arrays():
    x_dict = {
        'a': np.array([1]),
        'b': np.array([2]),
        'c': np.array([3]),
    }
    y_series = pd.Series(
        data=np.array([1, 2, 3]),
        index=['a', 'b', 'c'],
    )

    x_arr, y_arr, _ = blueno.preprocessing.to_arrays(x_dict, y_series)
    assert np.all(x_arr == np.expand_dims(y_arr, axis=1))


def test_to_arrays_sorted():
    x_dict = {
        'a': np.array([1]),
        'b': np.array([2]),
        'c': np.array([3]),
    }
    y_series = pd.Series(
        data=np.array([2, 1, 3]),
        index=['b', 'a', 'c'],
    )

    ids = blueno.preprocessing.to_arrays(x_dict, y_series)[2]
    assert np.all(ids == np.array(['a', 'b', 'c']))


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_prepare_data_correct_dims():
    params = {
        'data': blueno.DataConfig(**{
            'data_dir': '/home/lzhu7/elvo-analysis/data/'
                        'processed-standard/arrays/',
            'labels_path': '/home/lzhu7/elvo-analysis/data/'
                           'processed-standard/labels.csv',
            'index_col': 'Anon ID',
            'label_col': 'occlusion_exists',
            'gcs_url': 'gs://elvos/processed/processed-standard',
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,

        'generator': blueno.GeneratorConfig(
            generator_callable=lambda: None),

        'model': blueno.ModelConfig(**{
            # The callable must take in **kwargs as an argument
            'model_callable': small_model,
            'dropout_rate1': 0.8,
            'dropout_rate2': 0.7,
            'optimizer': keras.optimizers.Adam(lr=1e-4),
            'loss': keras.losses.categorical_crossentropy,
        }),
    }
    params = blueno.ParamConfig(**params)
    _, _, y_train, y_test, _, _ = preprocessing.prepare_data(
        params, train_test_val=False)
    assert y_train.ndim == 2
    assert y_test.ndim == 2


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_prepare_data_matching_indices():
    params = {
        'data': blueno.DataConfig(**{
            'data_dir': '/home/lzhu7/elvo-analysis/data/'
                        'processed-standard/arrays/',
            'labels_path': '/home/lzhu7/elvo-analysis/data/'
                           'processed-standard/labels.csv',
            'index_col': 'Anon ID',
            'label_col': 'occlusion_exists',
            'gcs_url': 'gs://elvos/processed/processed-standard'
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,
        'max_epochs': 1,

        'generator': blueno.GeneratorConfig(
            generator_callable=lambda: None),

        'model': blueno.ModelConfig(**{
            # The callable must take in **kwargs as an argument
            'model_callable': small_model,
            'dropout_rate1': 0.8,
            'dropout_rate2': 0.7,
            'optimizer': keras.optimizers.Adam(lr=1e-4),
            'loss': keras.losses.categorical_crossentropy,
        }),
    }
    params = blueno.ParamConfig(**params)
    _, _, y_train, y_test, id_train, id_test = preprocessing.prepare_data(
        params, train_test_val=False)
    for i, id_ in enumerate(id_test):
        if id_ == '068WBWCQGW5JHBYV':
            assert y_test[i][0] == 1
        elif id_ == 'FBGMN3O08GW5GG91':
            assert y_test[i][1] == 1
