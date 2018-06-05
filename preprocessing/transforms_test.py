from preprocessing import transforms
from preprocessing import parsers

def test_get_pixels_hu():
    print("hi")
    dicom_scan1 = parsers.load_scan(
        "/Users/haltriedman/PycharmProjects/elvo-analysis/data/AGTRMFUZM2MQUAB4 ALOE GABRIEL T")
    image1 = transforms.get_pixels_hu(dicom_scan1)
    print(image1.shape)
    assert image1

    dicom_scan2 = parsers.load_scan(
        "/Users/haltriedman/PycharmProjects/elvo-analysis/data/AHNGBFJSTEBO4BJW ADAS HORACE N")
    image2 = transforms.get_pixels_hu(dicom_scan2)
    print(image2.shape)

def main():
    test_get_pixels_hu()

if __name__ == "__main__":
    main()