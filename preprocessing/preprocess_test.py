import pytest as pytest

import preprocess


@pytest.skip()
def test_process_cab():
    arr = preprocess._process_cab(
        '/Users/lukezhu/Desktop/research/elvoai/data/RITRAQNET_10.137.213.144_20180531151240195/1.3.12.2.1107.5.1.4.66457.30000018012411161725100018246')
    print(arr.shape)
