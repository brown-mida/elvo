import datetime
import pathlib

import keras
import numpy as np
import os
import pytest

from .gcs import equal_array_counts, upload_gcs_plots

os.environ['CUDA_VISIBLE_DEVICES'] = ''


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_compare_dir_len():
    # ls processed-lower/arrays | wc -l
    # gsutil ls gs://elvos/processed/processed-lower/arrays | wc -l
    data_dir = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/'
                            'data/processed-lower')
    arrays_dir = data_dir / 'arrays'
    array_url = 'gs://elvos/processed/processed-lower/arrays'
    assert equal_array_counts(arrays_dir, array_url)


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_gcs_plots():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(1, activation='softmax'),
    ])
    model.compile(optimizer='sgd', loss='binary_crossentropy')

    x = np.random.rand(10, 224, 224, 3)
    y = np.random.randint(0, 2, size=(10, 1))
    history = model.fit(x, y)

    upload_gcs_plots(x,
                     x,
                     y,
                     model,
                     history,
                     'test_gcs',
                     datetime.datetime.utcnow().isoformat(),
                     plot_dir=pathlib.Path('test_gcs'))
