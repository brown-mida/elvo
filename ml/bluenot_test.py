import glob

import keras
import numpy as np
import os
import pytest
from elasticsearch_dsl import connections

import blueno
import bluenot
import generators.luke

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def setup_module():
    connections.create_connection(hosts=['http://104.196.51.205'])


def teardown_module():
    connections.remove_connection('default')


def small_model(input_shape=(224, 224, 3),
                num_classes=1,
                *args, **kwargs):
    return keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])


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
            'gcs_url': 'gs://elvos/processed/processed-standard'
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,
        'max_epochs': 1,

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
    bluenot.start_job(x_train, y_train, x_valid, y_valid, job_name='test_job',
                      username='test', params=params)


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
            'gcs_url': 'gs://elvos/processed/processed-standard',
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,
        'max_epochs': 1,

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
    bluenot.start_job(x_train, y_train, x_valid, y_valid, job_name='test_job',
                      username='test', params=params,
                      log_dir='/tmp/')
    for filepath in glob.glob('/tmp/test_job*'):
        os.remove(filepath)


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
            'label_col': 'occlusion_exists',
            'gcs_url': 'gs://elvos/processed/processed-standard',
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,
        'max_epochs': 1,

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
    x_train, x_valid, y_train, y_valid, _, _ = \
        bluenot.preprocessing.prepare_data(params, train_test_val=False)

    bluenot.start_job(x_train, y_train, x_valid, y_valid,
                      job_name='test_prepare_and_job', username='test',
                      params=params)


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_check_data_in_sync():
    params = {
        'data': blueno.DataConfig(**{
            'data_dir': '/home/lzhu7/elvo-analysis/data/'
                        'processed-standard/arrays/',
            'labels_path': '/home/lzhu7/elvo-analysis/data/'
                           'processed-standard/labels.csv',
            'index_col': 'Anon ID',
            'label_col': 'occlusion_exists',
            'gcs_url': 'gs://elvos/processed/processed-standard/',
        }),

        'val_split': 0.2,
        'seed': 0,
        'batch_size': 8,
        'max_epochs': 1,

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

    bluenot.check_data_in_sync(params)


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_check_data_in_sync_raises():
    with pytest.raises(ValueError):
        params = {
            'data': blueno.DataConfig(**{
                'data_dir': '/home/lzhu7/elvo-analysis/data/'
                            'processed-standard/arrays/',
                'labels_path': '/home/lzhu7/elvo-analysis/data/'
                               'processed-standard/labels.csv',
                'index_col': 'Anon ID',
                'label_col': 'occlusion_exists',
                'gcs_url': 'gs://elvos/processed/processed',
            }),

            'val_split': 0.2,
            'seed': 0,
            'batch_size': 8,
            'max_epochs': 1,

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

        bluenot.check_data_in_sync(params)
