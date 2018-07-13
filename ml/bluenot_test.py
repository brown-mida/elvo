import logging

import keras
import numpy as np
import os
import pandas as pd
import pytest

import blueno
import bluenot
import generators.luke


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

    x_arr, y_arr = bluenot.to_arrays(x_dict, y_series)
    print(y_arr)
    assert np.all(x_arr == np.expand_dims(y_arr, axis=1))


def test_start_job_no_err():
    x_train = np.random.uniform(0, 255, (100, 220, 220, 3))
    y_train = np.random.randint(0, 2, (100, 5))
    x_valid = np.random.uniform(0, 255, (20, 220, 220, 3))
    y_valid = np.random.randint(0, 2, (20, 5))
    params = {
        'data': blueno.DataConfig(**{
            'data_dir': '/home/lzhu7/elvo-analysis/data/'
                        'processed-standard/arrays/',
            'labels_path': '/home/lzhu7/elvo-analysis/data/'
                           'processed-standard/labels.csv',
            'index_col': 'Anon ID',
            'label_col': 'Location of occlusions on CTA (Matt verified)',
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,

        'generator': blueno.GeneratorConfig(
            generator_callable=generators.luke.standard_generators),

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
    logging.basicConfig(level=logging.DEBUG)
    bluenot.start_job(x_train, y_train, x_valid, y_valid,
                      job_name='test_job',
                      username='test',
                      params=params,
                      epochs=1)


def test_start_job_log():
    x_train = np.random.uniform(0, 255, (100, 220, 220, 3))
    y_train = np.random.randint(0, 2, (100, 5))
    x_valid = np.random.uniform(0, 255, (20, 220, 220, 3))
    y_valid = np.random.randint(0, 2, (20, 5))
    params = {
        'data': blueno.DataConfig(**{
            'data_dir': '/home/lzhu7/elvo-analysis/data/'
                        'processed-standard/arrays/',
            'labels_path': '/home/lzhu7/elvo-analysis/data/'
                           'processed-standard/labels.csv',
            'index_col': 'Anon ID',
            'label_col': 'Location of occlusions on CTA (Matt verified)',
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,

        'generator': blueno.GeneratorConfig(
            generator_callable=generators.luke.standard_generators),

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
    logging.basicConfig(level=logging.DEBUG)
    bluenot.start_job(x_train, y_train, x_valid, y_valid,
                      job_name='test_job',
                      username='test',
                      params=params,
                      epochs=1,
                      log_dir='/tmp/')


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
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,

        'generator': blueno.GeneratorConfig(
            generator_callable=generators.luke.standard_generators),

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
    _, _, y_train, y_test = bluenot.prepare_data(params)
    assert y_train.ndim == 2
    assert y_test.ndim == 2


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_prepare_and_job():
    params = {
        'data': blueno.DataConfig(**{
            'data_dir': '/home/lzhu7/elvo-analysis/data/'
                        'processed-standard/arrays/',
            'labels_path': '/home/lzhu7/elvo-analysis/data/'
                           'processed-standard/labels.csv',
            'index_col': 'Anon ID',
            'label_col': 'Location of occlusions on CTA (Matt verified)',
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,

        'generator': blueno.GeneratorConfig(
            generator_callable=generators.luke.standard_generators),

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
    x_train, x_valid, y_train, y_valid = bluenot.prepare_data(params)
    bluenot.start_job(x_train,
                      y_train,
                      x_valid,
                      y_valid,
                      job_name='test_prepare_and_job',
                      username='test',
                      params=params,
                      epochs=1)
