import pathlib

import os
import pytest

from blueno import reporting


def test_parse_filename():
    filename = 'processed-lower_1-classes-2018-07-10T03:22:18.003758.csv'
    actual = reporting._parse_filename(filename)
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

    assert reporting._extract_params(filepath) == expected


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_extract_metrics():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'data_2-classes-2018-07-07T10:22:47.601628.csv')
    expected = reporting.Metrics(epochs=99, train_acc=0.8790123458261843,
                                 final_val_acc=0.7044334990050405,
                                 best_val_acc=0.7192118223664796,
                                 final_val_loss=0.7228950116728327,
                                 best_val_loss=0.5968567865529084,
                                 final_val_sensitivity=0.7044334990050405,
                                 best_val_sensitivity=0.7192118223664796)

    assert reporting._extract_metrics(path) == expected


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_extract_author():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'test_prepare_and_job-2018-07-12T03:09:27.805668.log')
    assert reporting._extract_author(path) == 'sumera'


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_ended_at():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'test_job-2018-07-12T03:17:29.021608.log')
    assert reporting._extract_ended_at(
        path) == '2018-07-12T03:19:46.868800'


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_bluenom_extract_model_url():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'test_job-2018-07-12T03:17:29.021608.log')
    assert reporting._extract_model_url(path) is None


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_bluenom_extract_auc():
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs/'
                        'processed-no-basvert_2-classes'
                        '-2018-07-13T15:26:36.405530.log')
    assert reporting._extract_auc(path) == 0.7974631751227496
