import pytest as pytest


@pytest.mark.skip(reason='Requires a file')
def test_process_cab():
    # preprocess.py is a script so we can't import it without path isses
    import preprocess
    filepath = '/Users/lukezhu/Desktop/research/elvoai/data/' \
               'RITRAQNET_10.137.213.144_20180531151240195/' \
               '1.3.12.2.1107.5.1.4.66457.30000018012411161725100018246'
    arr = preprocess._process_cab(filepath)
    print(arr.shape)
