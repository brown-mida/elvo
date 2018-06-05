from preprocessing import transforms
from preprocessing import parsers
from pathlib import Path

DATA_DIR = '../data'

def test_get_pixels_hu():
    rel_path = 'AGTRMFUZM2MQUAB4 ALOE GABRIEL T/' \
               '642d0e0395789b1d78fe8cb82650f238 CTA ELVO Head and Neck/' \
               'CT axial brain neck cta'
    abs_path = Path(DATA_DIR) / rel_path

    dicom_scan = parsers.load_scan(abs_path)
    image = transforms.get_pixels_hu(dicom_scan)
    print(image.shape)
