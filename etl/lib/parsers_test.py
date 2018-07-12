from pathlib import Path

from lib import parsers

DATA_DIR = 'data'


def test_load_scan_no_error():
    rel_path = 'ABCDE SAMPLE'
    abs_path = Path(DATA_DIR) / rel_path
    assert parsers.load_scan(str(abs_path)) is not None


def test_load_patient_infos():
    path = str(Path(DATA_DIR))
    rel_path_1 = 'ABCDE SAMPLE'
    abs_path_1 = Path(DATA_DIR) / rel_path_1

    rel_path_2 = 'DEFGH SAMPLE2'
    abs_path_2 = Path(DATA_DIR) / rel_path_2

    result = parsers.load_patient_infos(path)
    assert isinstance(result, dict)
    assert result['ABCDE'] == str(abs_path_1)
    assert result['DEFGH'] == str(abs_path_2)
