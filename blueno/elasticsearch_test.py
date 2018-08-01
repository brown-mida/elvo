import pathlib

import os
import pytest
from elasticsearch_dsl import connections

from blueno import elasticsearch


def setup_module():
    connections.create_connection(hosts=['http://104.196.51.205'])


def teardown_module():
    connections.remove_connection('default')


def test_parse_filename():
    filename = 'processed-lower_1-classes-2018-07-10T03:22:18.003758.csv'
    actual = elasticsearch._parse_filename(filename)
    assert actual == ('processed-lower_1-classes',
                      '2018-07-10T03:22:18.003758')


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_extract_params():
    filepath = pathlib.Path(
        '/gpfs/main/home/lzhu7/elvo-analysis/logs/'
        'processed-lower_2-classes-2018-07-10T18:17:13.665080.log'
    )

    expected = "{'val_split': 0.2, 'seed': 42," \
               " 'model': {'rotation_range': 30," \
               " 'optimizer':" \
               " <keras.optimizers.SGD object at 0x7ff0945d1da0>," \
               " 'model_callable': <function resnet at 0x7ff0945a8f28>," \
               " 'loss':" \
               " <function categorical_crossentropy at 0x7ff0946a57b8>," \
               " 'freeze': False," \
               " 'dropout_rate2': 0.8, 'dropout_rate1': 0.8," \
               " 'batch_size': 8}," \
               " 'generator':" \
               " <function standard_generators at 0x7ff0946137b8>," \
               " 'data': {'data_dir':" \
               " '/home/lzhu7/elvo-analysis/data/processed-lower/arrays/'," \
               " 'labels_path':" \
               " '/home/lzhu7/elvo-analysis/data/" \
               "processed-lower/labels.csv'," \
               " 'index_col': 'Anon ID', 'label_col': 'occlusion_exists'}}\n"

    assert elasticsearch._extract_params(filepath) == expected


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_extract_metrics():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'data_2-classes-2018-07-07T10:22:47.601628.csv')
    expected = elasticsearch.Metrics(epochs=99, train_acc=0.8790123458261843,
                                     final_val_acc=0.7044334990050405,
                                     best_val_acc=0.7192118223664796,
                                     final_val_loss=0.7228950116728327,
                                     best_val_loss=0.5968567865529084,
                                     final_val_sensitivity=0.7044334990050405,
                                     best_val_sensitivity=0.7192118223664796,
                                     final_val_specificity=0.7044334990050405,
                                     best_val_specificity=0.7192118223664796)

    assert elasticsearch._extract_metrics(path) == expected


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_extract_author():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'test_prepare_and_job-2018-07-12T03:09:27.805668.log')
    assert elasticsearch._extract_author(path) == 'sumera'


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_ended_at():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'test_job-2018-07-12T03:17:29.021608.log')
    assert elasticsearch._extract_ended_at(
        path) == '2018-07-12T03:19:46.868800'


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_bluenom_extract_model_url():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'test_job-2018-07-12T03:17:29.021608.log')
    assert elasticsearch._extract_model_url(path) is None


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_bluenom_extract_auc():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'processed-no-basvert_2-classes'
                        '-2018-07-13T15:26:36.405530.log')
    assert elasticsearch._extract_auc(path) == 0.7974631751227496


def test_parse_params_str_config():
    params_str = ("ParamConfig(data=DataConfig("
                  "data_dir='/home/lzhu7/elvo-analysis/data/"
                  "processed-lower-nbv/arrays', labels_path='/home/lzhu7/"
                  "elvo-analysis/data/processed-lower-nbv/labels.csv',"
                  " index_col='Anon ID', label_col='occlusion_exists'),"
                  " generator=GeneratorConfig(generator_callable="
                  "<function standard_generators at 0x7f986e78a598>,"
                  " rotation_range=20, width_shift_range=0.1,"
                  " height_shift_range=0.1, shear_range=0.2, zoom_range=0.1,"
                  " horizontal_flip=True, vertical_flip=False),"
                  " model=ModelConfig(model_callable="
                  "<function resnet at 0x7f986e71fd90>,"
                  " optimizer="
                  "<keras.optimizers.Adam object at 0x7f986e77e668>,"
                  " loss=<function binary_crossentropy at 0x7f9874f1ea60>,"
                  " dropout_rate1=0.7, dropout_rate2=0.7, freeze=False),"
                  " batch_size=3, seed=0, val_split=0.1, job_fn=None)")
    expected = {
        'batch_size': 3, 'val_split': 0.1, 'seed': 0, 'rotation_range': 20,
        'width_shift_range': 0.1, 'height_shift_range': 0.1,
        'shear_range': 0.2, 'horizontal_flip': True,
        'vertical_flip': False, 'dropout_rate1': 0.7,
        'dropout_rate2': 0.7,
        'data_dir': "/home/lzhu7/elvo-analysis/data/"
                    "processed-lower-nbv/arrays"
    }
    assert elasticsearch._parse_params_str(params_str) == expected


def test_parse_params_val_split():
    params_str = ("ParamConfig(data=DataConfig("
                  "data_dir='/home/lzhu7/elvo-analysis/data/"
                  "processed-lower-nbv/arrays', labels_path='/home/lzhu7/"
                  "elvo-analysis/data/processed-lower-nbv/labels.csv',"
                  " index_col='Anon ID', label_col='occlusion_exists'),"
                  " generator=GeneratorConfig(generator_callable="
                  "<function standard_generators at 0x7f986e78a598>,"
                  " rotation_range=20, width_shift_range=0.1,"
                  " height_shift_range=0.1, shear_range=0.2,"
                  " zoom_range=(0.9, 1.1),"
                  " horizontal_flip=True, vertical_flip=False),"
                  " model=ModelConfig(model_callable="
                  "<function resnet at 0x7f986e71fd90>,"
                  " optimizer="
                  "<keras.optimizers.Adam object at 0x7f986e77e668>,"
                  " loss=<function binary_crossentropy at 0x7f9874f1ea60>,"
                  " dropout_rate1=0.7, dropout_rate2=0.7, freeze=False),"
                  " batch_size=3, seed=0, val_split=0.1, job_fn=None)")
    expected = {
        'batch_size': 3, 'val_split': 0.1, 'seed': 0, 'rotation_range': 20,
        'width_shift_range': 0.1, 'height_shift_range': 0.1,
        'shear_range': 0.2, 'horizontal_flip': True,
        'vertical_flip': False, 'dropout_rate1': 0.7,
        'dropout_rate2': 0.7,
        'data_dir': "/home/lzhu7/elvo-analysis/data/"
                    "processed-lower-nbv/arrays",
        'zoom_range': '(0.9, 1.1)',
    }
    assert elasticsearch._parse_params_str(params_str) == expected


def test_parse_params_str_dict():
    params_str = "{'val_split': 0.2, 'seed': 42," \
                 " 'model': {'rotation_range': 20, 'optimizer':" \
                 " <keras.optimizers.Adam object at 0x7f2fead000b8>," \
                 " 'model_callable': <function resnet at 0x7f2fae954f28>," \
                 " 'loss': <function binary_crossentropy at 0x7f2faea508c8>," \
                 " 'freeze': True, 'dropout_rate2': 0.8," \
                 " 'dropout_rate1': 0.8," \
                 " 'batch_size': 8}, 'generator':" \
                 " <function standard_generators at 0x7f2fae9c07b8>," \
                 " 'data': {'data_dir':" \
                 " '/home/lzhu7/elvo-analysis/data/" \
                 "processed-standard/arrays/'," \
                 " 'labels_path': '/home/lzhu7/elvo-analysis/data/" \
                 "processed-standard/labels.csv'," \
                 " 'index_col': 'Anon ID', 'label_col': 'occlusion_exists'}}"
    expected = {
        'batch_size': 8, 'val_split': 0.2, 'seed': 42, 'rotation_range': 20,
        'dropout_rate1': 0.8, 'dropout_rate2': 0.8,
        'data_dir': "/home/lzhu7/elvo-analysis/data/"
                    "processed-standard/arrays/"
    }
    assert elasticsearch._parse_params_str(params_str) == expected


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses logs only on gpu1708')
def test_extract_best_auc():
    log_filepath = pathlib.Path(
        '/gpfs/main/home/lzhu7/elvo-analysis/logs/'
        'processed-only-m1_2-classes-2018-07-18T15:47:55.783659.log'
    )

    assert elasticsearch._extract_best_auc(log_filepath) == 0.9267489711934156


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_insert_or_ignore():
    metrics_file = pathlib.Path(
        '/gpfs/main/home/lzhu7/elvo-analysis/logs/'
        'processed-only-m1_2-classes-2018-07-18T15:47:54.618535.csv')
    log_file = pathlib.Path(
        '/gpfs/main/home/lzhu7/elvo-analysis/logs/'
        'processed-only-m1_2-classes-2018-07-18T15:47:54.618535.log')
    elasticsearch.insert_or_ignore_filepaths(log_file,
                                             metrics_file)


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_insert_or_update():
    metrics_file = pathlib.Path(
        '/gpfs/main/home/lzhu7/elvo-analysis/logs/'
        'processed-only-m1_2-classes-2018-07-18T15:47:54.618535.csv')
    log_file = pathlib.Path(
        '/gpfs/main/home/lzhu7/elvo-analysis/logs/'
        'processed-only-m1_2-classes-2018-07-18T15:47:54.618535.log')
    elasticsearch.insert_or_replace_filepaths(log_file,
                                              metrics_file)
