import transforms


def test_parse_bounding_box():
    # TODO(luke): Make sure that this path works on Windows
    path = 'RI Hospital ELVO Data/51456445/AnnotationROI.acsv'
    point = transforms.parse_bounding_box(path)
    assert point == (
        (-14.7493, 177.584, -91.7008),
        (20.0315, 8.17094, 8.05191),
    )
