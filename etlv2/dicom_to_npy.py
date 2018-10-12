"""
Methods for converting DICOM files to npy files.
"""
import io
import logging
import numpy as np
import os
import pydicom
import shutil
import subprocess
import time
import traceback
from google.cloud import storage
from typing import List, Dict

from scipy.ndimage import zoom


def process_cab(blob: storage.Blob, patient_id: str) -> np.ndarray:
    """
    Downloads the blob and return a 3D standardized numpy array.

    :param blob:
    :param patient_id:
    :return:
    """
    # TODO: Fix issues with process_cab workingdir failing
    old_wd = os.getcwd()
    dirname = f'/tmp/dicom_to_npy-{int(time.time())}'
    os.makedirs(dirname, exist_ok=True)
    os.chdir(dirname)

    blob.download_to_filename(patient_id + '.cab')
    subprocess.call(['cabextract', patient_id + '.cab'])
    logging.info('extracted cab file')

    dirpath = list(os.walk('.'))[2][0]
    logging.info(f'loading scans from {dirpath}')
    processed_scan = _process_cab(dirpath)

    os.chdir(old_wd)
    shutil.rmtree(dirname)
    return processed_scan


def _process_cab(dirpath: str) -> np.array:
    scan = load_scan(dirpath)
    processed_scan = preprocess_scan(scan)
    return processed_scan


def process_zip(blob: storage.Blob, patient_id: str) -> np.ndarray:
    """
    Downloads the blob and returns a 3D standardized numpy array.
    :param blob:
    :param patient_id:
    :return:
    """
    old_wd = os.getcwd()
    dirname = f'/tmp/dicom_to_npy-{int(time.time())}'
    os.makedirs(dirname, exist_ok=True)
    os.chdir(dirname)

    blob.download_to_filename(patient_id + '.zip')
    logging.info('extracting zip file')
    shutil.unpack_archive(patient_id + '.zip', format='zip')

    dirpath = list(os.walk('.'))[3][0]
    logging.info(f'loading scans from {dirpath}')
    scan = load_scan(dirpath)
    processed_scan = preprocess_scan(scan)
    logging.info(f'processing dicom data')

    os.chdir(old_wd)
    shutil.rmtree(dirname)
    return processed_scan


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
        ELVOS_anon/HIA2VPHI6ABMCQTV HANKERSON IGNACIO A/f8...
    to its patient id: HIA2VPHI6ABMCQTV
    """
    return dirpath[len(input_dir) + 1:].split()[0]


def save_to_gcs(processed_scan, outpath, bucket):
    stream = io.BytesIO()
    np.save(stream, processed_scan)
    processed_blob = storage.Blob(outpath, bucket=bucket)
    stream.seek(0)
    processed_blob.upload_from_file(stream)
    logging.info(f'saving dicom data to GCS in {outpath}')
    stream.close()


def preprocess_scan(slices: List[pydicom.FileDataset]) -> np.array:
    """Transforms the input dicom slices into a numpy array of pixels
    in Hounsfield units with standardized spacing.
    """
    scan = get_pixels_hu(slices)
    scan = standardize_spacing(scan, slices)
    return scan


def get_pixels_hu(slices):
    """
    Takes in a list of dicom datasets and returns the 3D pixel array in
    Hounsfield scale, taking slope and intercept into account.

    :param slices: 3D DICOM image to process
    :return:
    """
    for s in slices:
        assert s.Rows == 512
        assert s.Columns == 512

    image = np.stack([np.frombuffer(s.pixel_array, np.int16).reshape(512, 512)
                      for s in slices])

    # Convert the pixels Hounsfield units (HU)
    intercept = 0
    for i, s in enumerate(slices):
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope
        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)
        image[i] += np.int16(intercept)

    # Some scans use -2000 as the default value for pixels not in the body
    # We set these pixels to -1000, the HU for air
    image[image == intercept - 2000] = -1000

    return image


def standardize_spacing(image, slices):
    """
    Takes in a 3D image and interpolates the image so each pixel corresponds to
    approximately a 1x1x1 box.

    :param image: Non-interpolated array
    :param slices: actual DICOM slices with spacing info
    :return:
    """
    # Determine current pixel spacing
    spacing = np.array(
        [slices[0].SliceThickness] + list(slices[0].PixelSpacing),
        dtype=np.float32
    )
    new_shape = np.round(image.shape * spacing)
    resize_factor = new_shape / image.shape

    # zoom to the right size
    return zoom(image, resize_factor, mode='nearest')


def up_to_date(input_blob: storage.Blob, output_blob: storage.Blob):
    """
    Checks if the blob is up-to-date.

    :param input_blob:
    :param output_blob:
    :return: true if the output blob is up-to-date. If the blob doesn't
    exist or is outdated, returns false.
    """
    if not output_blob.exists():
        return False

    input_blob.reload()
    output_blob.reload()
    assert input_blob.updated is not None, 'input blob should exist'
    if input_blob.updated > output_blob.updated:
        return False

    return True


def dicom_to_npy(in_dir: str, out_dir: str):
    """
    :param in_dir: directory in gs://elvos to load from. must end with /
    :param out_dir: directory in gs://elvos to save to. must end with /
    :return:
    """
    gcs_client = storage.Client(project='elvo-198322')
    bucket = gcs_client.get_bucket('elvos')

    blob: storage.Blob
    for blob in bucket.list_blobs(prefix=in_dir):
        try:
            if len(blob.name) < 4 or blob.name[-4:] not in ('.zip', '.cab'):
                logging.info(f'ignoring non-data file {blob.name}')
                continue

            logging.info(f'processing blob {blob.name}')
            patient_id = blob.name[len(in_dir): -len('.cab')]
            outpath = f'{out_dir}{patient_id}.npy'

            if up_to_date(blob, storage.Blob(outpath, bucket)):
                logging.info(f'outfile {outpath} already exists')
                continue

            logging.info(f'outfile {outpath} is outdated, updating')
            if blob.name.endswith('.cab'):
                processed_scan = process_cab(blob, patient_id)
                save_to_gcs(processed_scan, outpath, bucket)
            elif blob.name.endswith('.zip'):
                processed_scan = process_zip(blob, patient_id)
                save_to_gcs(processed_scan, outpath, bucket)
            else:
                logging.info(f'file extension must be .cab or .zip,'
                             f' got {blob.name}')
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())


if __name__ == '__main__':
    logging.basicConfig()
    dicom_to_npy('ELVOs_anon/', 'raw_numpy/')
