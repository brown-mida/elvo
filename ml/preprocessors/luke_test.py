import os
import pytest

import preprocessors.luke


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_preprocess():
    data_dir = '/home/lzhu7/elvo-analysis/data/numpy_compressed/'
    labels_dir = '/home/lzhu7/elvo-analysis/data/metadata/'
    arrays, labels = preprocessors.luke.preprocess_data(data_dir,
                                                        labels_dir,
                                                        30,
                                                        24,
                                                        (0, 400),
                                                        test=True)
    for arr in arrays.values():
        assert arr.min() >= 0
        assert arr.max() <= 400
