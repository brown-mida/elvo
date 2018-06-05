from pathlib import Path

from preprocessing import parsers

DATA_DIR = '../data'


def test_load_scan_no_error():
    rel_path = 'AGTRMFUZM2MQUAB4 ALOE GABRIEL T/' \
               '642d0e0395789b1d78fe8cb82650f238 CTA ELVO Head and Neck/' \
               'CT axial brain neck cta'
    abs_path = Path(DATA_DIR) / rel_path
    assert parsers.load_scan(str(abs_path)) is not None
