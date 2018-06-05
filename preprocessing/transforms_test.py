from preprocessing import transforms
from preprocessing import parsers
from pathlib import Path
from matplotlib import pyplot as plt

DATA_DIR = "../data"


def test_get_pixels_hu():
    # load the DICOM images and get the scan in an array format
    rel_path = 'AGTRMFUZM2MQUAB4 ALOE GABRIEL T/' \
               '642d0e0395789b1d78fe8cb82650f238 CTA ELVO Head and Neck/' \
               'CT axial brain neck cta'
    abs_path = Path(DATA_DIR) / rel_path
    dicom_scan = parsers.load_scan(str(abs_path))

    # put the pixels into Hounsfield units and make sure the image has been properly resized
    image = transforms.get_pixels_hu(dicom_scan)

    # show random slice of the Hounsfield image
    plt.imshow(image[200])
    plt.show()
    assert image.shape == (350, 512, 512)


def test_standarize_spacing():
    # load the DICOM images and get the scan in an array format
    rel_path = 'AHNGBFJSTEBO4BJW ADAS HORACE N/' \
                '5ae5ca5c99037e80991199f75c7a5002 CTA ELVO Head and Neck/' \
                'CT CTA Head and Neck'
    abs_path = Path(DATA_DIR) / rel_path
    dicom_scan = parsers.load_scan(str(abs_path))

    # put the pixels into Hounsfield units and resize the image
    image = transforms.get_pixels_hu(dicom_scan)

    # use standardize_spacing to shrink the image and interpolate it to roughly 1x1x1 mm^3 cubes
    standardized_img = transforms.standardize_spacing(image, dicom_scan)

    # show random slice of the standardized image
    plt.imshow(standardized_img[100])
    plt.show()
    assert image.shape[1] > standardized_img.shape[1]
    assert image.shape[2] > standardized_img.shape[2]


def test_crop():
    # load the DICOM images and get the scan in an array format
    rel_path = 'AGTRMFUZM2MQUAB4 ALOE GABRIEL T/' \
               '642d0e0395789b1d78fe8cb82650f238 CTA ELVO Head and Neck/' \
               'CT axial brain neck cta'
    abs_path = Path(DATA_DIR) / rel_path
    dicom_scan = parsers.load_scan(str(abs_path))

    # put the pixels into Hounsfield units and resize the image
    image = transforms.get_pixels_hu(dicom_scan)

    # use standardize_spacing to shrink the image and interpolate it to roughly 1x1x1 mm^3 cubes
    standardized_img = transforms.standardize_spacing(image, dicom_scan)

    # crop to 200x200x200 and show cropped image
    cropped_img = transforms.crop(standardized_img)
    plt.imshow(cropped_img[100])
    plt.show()
    assert cropped_img.shape == (200, 200, 200)
