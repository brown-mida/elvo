from pathlib import Path

from preprocessing import parsers

DATA_DIR = '../data'


def test_load_scan_no_error():
    rel_path = 'AGTRMFUZM2MQUAB4 ALOE GABRIEL T/' \
               '642d0e0395789b1d78fe8cb82650f238 CTA ELVO Head and Neck/' \
               'CT axial brain neck cta'
    abs_path = Path(DATA_DIR) / rel_path
    assert parsers.load_scan(str(abs_path)) is not None

def test_load_patient_infos():
   path = str(Path(DATA_DIR))
   rel_path_1 = 'AGTRMFUZM2MQUAB4 ALOE GABRIEL T/' \
                '642d0e0395789b1d78fe8cb82650f238 CTA ELVO Head and Neck/' \
                'CT axial brain neck cta'
   abs_path_1 = Path(DATA_DIR) / rel_path_1

   rel_path_2 = 'AHNGBFJSTEBO4BJW ADAS HORACE N/'\
                '5ae5ca5c99037e80991199f75c7a5002 CTA ELVO Head and Neck/' \
                'CT CTA Head and Neck'
   abs_path_2 = Path(DATA_DIR) / rel_path_2

   result = parsers.load_patient_infos(path)
   assert isinstance(result, dict)
   assert result['AGTRMFUZM2MQUAB4'] == str(abs_path_1)
   assert result['AHNGBFJSTEBO4BJW'] == str(abs_path_2)