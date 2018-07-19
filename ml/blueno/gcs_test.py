import pathlib

import os
import pytest

from .gcs import equal_array_counts


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_compare_dir_len():
    # ls processed-lower/arrays | wc -l
    # gsutil ls gs://elvos/processed/processed-lower/arrays | wc -l
    data_dir = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/'
                            'data/processed-lower')
    arrays_dir = data_dir / 'arrays'
    array_url = 'gs://elvos/processed/processed-lower/arrays'
    assert equal_array_counts(arrays_dir, array_url) == True
