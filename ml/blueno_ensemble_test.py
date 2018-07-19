import pytest

import blueno_ensemble


@pytest.mark.skip(reason='Not a test and takes too long')
def test_bluenoe():
    blueno_ensemble.ensemble()


def test_parse_filename():
    filename = 'processed-lower_2-classes-2018-07-13T09:59:19.643349.hdf5'
    assert blueno_ensemble._parse_filename(filename) == (
        'processed-lower_2-classes',
        '2018-07-13T09:59:19.643349')
