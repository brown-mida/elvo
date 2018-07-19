import pytest

import bluenoe


@pytest.mark.skip(reason='Not a test and takes too long')
def test_bluenoe():
    bluenoe.ensemble()


def test_parse_filename():
    filename = 'processed-lower_2-classes-2018-07-13T09:59:19.643349.hdf5'
    assert bluenoe._parse_filename(filename) == ('processed-lower_2-classes',
                                                 '2018-07-13T09:59:19.643349')
