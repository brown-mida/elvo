import pathlib

from .gcs import equal_array_counts


def test_compare_dir_len():
    # ls processed-lower/arrays | wc -l
    # gsutil ls gs://elvos/processed/processed-lower/arrays | wc -l
    data_dir = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/'
                            'data/processed-lower')
    arrays_dir = data_dir / 'arrays'
    array_url = 'gs://elvos/processed/processed-lower/arrays'
    assert equal_array_counts(arrays_dir, array_url) == True
