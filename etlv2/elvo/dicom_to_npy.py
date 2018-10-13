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
    Loads and processes a .cab file on GCS into a 3D standardized numpy
    array.

    Note: This will only work on machines with cabextract installed.

    :param blob: the GCS object containing array data
    # TODO(luke): refactor param patient_id to out_filename
    :param patient_id: a legacy str used to specify what temp filenames
        to save the blob to.
    :return: the processed numpy array
    """
    old_wd = os.getcwd()
    dirname = f'/tmp/dicom_to_npy-{int(time.time())}'
    os.makedirs(dirname, exist_ok=True)
    os.chdir(dirname)

    blob.download_to_filename(patient_id + '.cab')
    # Make sure that cabextract is installed on the VM!
    subprocess.call(['cabextract', patient_id + '.cab'])
    logging.info('extracted cab file')

    dirpath = list(os.walk('.'))[2][0]
    logging.info(f'loading scans from {dirpath}')
    scan = load_scan(dirpath)
    processed_scan = preprocess_scan(scan)

    os.chdir(old_wd)
    shutil.rmtree(dirname)
    return processed_scan


def process_zip(blob: storage.Blob, patient_id: str) -> np.ndarray:
    """
    Loads and processes a .zip file on GCS into a 3D standardized numpy
    array.

    :param blob: the GCS object containing array data
    # TODO(luke): refactor param patient_id to out_filename
    :param patient_id: a legacy str used to specify what temp filenames
        to save the blob to.
    :return: the processed numpy array
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
    """
    Loads all DICOM files contained in dirpath, sorted by
    the ImagePositionPatient[2] DICOM field in ascending order

    This only loads files ending in .dcm and ignores nested
    folders.

    :param dirpath: the path to a directory containing dicom (.dcm)
        files
    :return: a list of all pydicom FileDataset objects, sorted by
        ImagePositionPatient
    """
    slices = [pydicom.read_file(dirpath + '/' + filename)
              for filename in os.listdir(dirpath)
              if filename.endswith('.dcm')]
    return sorted(slices, key=lambda x: float(x.ImagePositionPatient[2]))


def save_to_gcs(array: np.ndarray,
                blob_name: str,
                bucket: storage.Bucket) -> None:
    """
    Saves the array to GCS.
    
    :param array: the numpy array to save
    :param blob_name: the name of the output blob
    :param bucket: the output bucket
    :return:
    """
    stream = io.BytesIO()
    # noinspection PyTypeChecker
    np.save(stream, array)
    processed_blob = storage.Blob(blob_name, bucket=bucket)
    stream.seek(0)
    processed_blob.upload_from_file(stream)
    logging.info(f'saving dicom data to GCS in {blob_name}')
    stream.close()


def preprocess_scan(slices: List[pydicom.FileDataset]) -> np.array:
    """
    Transforms the input dicom slices into a numpy array of pixels
    in Hounsfield units with standardized spacing.

    :param slices: a list of sorted slices
    :return: a numpy array
    """
    # TODO(luke): Data validation w/ asserts
    scan = get_pixels_hu(slices)
    scan = standardize_spacing(scan, slices)
    return scan


def get_pixels_hu(slices: List[pydicom.FileDataset]) -> np.ndarray:
    """
    Processes the slices into a 3D Hounsfied unit pixel array.

    :param slices: sorted list of slices 3D DICOM image to process
    :return: a 3D numpy array
    """
    for s in slices:
        assert s.Rows == 512
        assert s.Columns == 512

    image = np.stack([np.frombuffer(s.pixel_array, np.int16).reshape(512, 512)
                      for s in slices])

    # Convert the pixels to Hounsfield units (HU)
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


def standardize_spacing(image, slices) -> np.ndarray:
    """
    Interpolates the image so each pixel corresponds to
    approximately a 1mm^3 box.

    :param image: input numpy array
    :param slices: actual DICOM slices with spacing info
    :return: a standardized array
    """
    # pixel spacing values in (height, length, width) order
    height_spacing = slices[0].SliceThickness
    length_spacing, width_spacing = slices[0].PixelSpacing

    for s in slices:
        assert s.SliceThickness == height_spacing
        assert s.PixelSpacing[0] == length_spacing
        assert s.PixelSpacing[1] == width_spacing

    spacing = np.array([height_spacing, length_spacing, width_spacing])
    new_shape = np.round(image.shape * spacing)
    resize_factor = new_shape / image.shape

    return zoom(image, resize_factor, mode='nearest')


def up_to_date(input_blob: storage.Blob, output_blob: storage.Blob):
    """
    Checks if the output blob is up-to-date with the input blob.

    :param input_blob:
    :param output_blob:
    :return: true if the output blob is exists and was updated after
    the input blob was last updated, false otherwise
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
    Loads .cab and .zip files in the in_dir and saves processed
    .npy data in the output directory.

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
            # TODO(luke): This should be removed. We don't want bad data
            logging.error(e)
            logging.error(traceback.format_exc())


if __name__ == '__main__':
    logging.basicConfig()
    dicom_to_npy('ELVOs_anon/', 'raw_numpy/')
