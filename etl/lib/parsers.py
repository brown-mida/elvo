"""
This is a set of utilities for parsing DICOM files (the standard file format
for medical imaging data).
"""

import logging
import os
import re
import shutil
from typing import List, Dict
import pydicom


def load_scans(input_dir: str):
    """
    Loads scans from an input directory

    :param input_dir: directory to load scans from
    :return: set of patient IDs and the preprocessed scans
    """
    id_pattern = re.compile(r'\d+')
    patient_ids = []
    preprocessed_scans = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if filenames and '.dcm' in filenames[0]:
            patient_id = id_pattern.findall(dirpath)[0]
            patient_ids.append(patient_id)
            preprocessed_scans.append(load_scan(dirpath))
    return patient_ids, preprocessed_scans


def load_scan(dirpath: str) -> List[pydicom.FileDataset]:
    """Loads a CT scan.

    This only loads dicom files ending in .dcm and ignores nested
    folders.

    :param dirpath: the path to a directory containing dicom (.dcm) files
    :return: a list of all dicom FileDataset objects, sorted by
        ImagePositionPatient
    """
    slices = [pydicom.read_file(dirpath + '/' + filename)
              for filename in os.listdir(dirpath)
              if filename.endswith('.dcm')]
    return sorted(slices, key=lambda x: float(x.ImagePositionPatient[2]))


def load_patient_infos(input_dir: str) -> Dict[str, str]:
    """Returns a mapping of patient ids to the directory of scans"""
    patient_ids = {}
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if filenames and '.dcm' in filenames[0]:
            patient_id = _parse_id(dirpath, input_dir)
            patient_ids[patient_id] = dirpath
    return patient_ids


def _parse_id(dirpath: str, input_dir: str) -> str:
    """Turns a dirpath like
        ELVOS/anon/HIA2VPHI6ABMCQTV HANKERSON IGNACIO A/f8...
    to its patient id: HIA2VPHI6ABMCQTV
    """
    return dirpath[len(input_dir) + 1:].split()[0]


def unzip_scans(input_dir: str, remove_zip=True):
    """Unzips the zip files in the directory, writing to folders in
    input_dir with the same name.
    """
    previous_dir = os.getcwd()
    os.chdir(input_dir)
    for filepath in os.listdir('.'):
        if filepath.endswith('.zip'):
            try:
                shutil.unpack_archive(filepath, format='zip')
                logging.debug(
                    'Unzipped data in file {}'.format(filepath)
                )
                if remove_zip:
                    os.remove(filepath)
            except OSError:
                logging.debug('Error unzipping {}'.format(filepath))
    os.chdir(previous_dir)


def parse_bounding_box(annotation_path: str):
    """Parses an AnnotationROI.acsv file and returns a tuple of
    the region of interest.

    The first triple in the tuple is the center of the ROI.
    The second triple in the tuple is the distance of the
    bounding box from the center.

    For example, the pair ((0, 0, 0), (1, 1, 1)) would described
    the box with coordinates (1, 1, 1), (-1, -1, -1), (-1, 1, 1) ...
    """
    point = []
    with open(annotation_path, 'r') as annotation_fp:
        for line in annotation_fp:
            if line.startswith('# pointNumberingScheme'):
                assert line == '# pointNumberingScheme = 0\n'
            if line.startswith('# pointColumns'):
                assert line == '# pointColumns = type|x|y|z|sel|vis\n'
            if line.startswith('point|'):
                values = line.split('|')
                coordinates = (
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                )
                point.append(coordinates)
    assert len(point) == 2
    return tuple(point)
