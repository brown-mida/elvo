"""To run this GOOGLE_APPLICATION_CREDENTIALS should be set.

This script converts the compressed patient info in ELVOs_anon
into numpy files, saved in the numpy folder of the elvo bucket.
"""
import io
import logging
import os
import shutil
import subprocess
from typing import List

import numpy as np
import pandas as pd
import pydicom
from google.cloud import storage

from lib import parsers
from lib import transforms

ELVOS_ANON = 'ELVOs_anon'
EXTENSION_LENGTH = len('.cab')  # == 4


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def load_metadata(bucket):
    """Loads the metadata from GCS.
    """
    positives_blob = bucket.get_blob('metadata/positives.csv')
    positives_bytes = positives_blob.download_as_string()
    positives_str = positives_bytes.decode('utf-8')
    positives_df = pd.read_csv(io.StringIO(positives_str))

    negatives_blob = bucket.get_blob('metadata/negatives.csv')
    negatives_bytes = negatives_blob.download_as_string()
    negatives_str = negatives_bytes.decode('utf-8')
    negatives_df = pd.read_csv(io.StringIO(negatives_str))
    return positives_df, negatives_df


def process_cab(blob: storage.Blob, patient_id: str) -> None:
    os.mkdir('tmp')
    os.chdir('tmp')

    blob.download_to_filename(patient_id + '.cab')
    subprocess.call(['cabextract', patient_id + '.cab'])
    logging.info('extracted cab file')

    dirpath = list(os.walk('.'))[2][0]
    logging.info(f'loading scans from {dirpath}')
    processed_scan = _process_cab(dirpath)

    os.chdir('..')
    shutil.rmtree('tmp')

    save_to_gcs(processed_scan, patient_id, blob.bucket)


def _process_cab(dirpath: str) -> np.array:
    scan = parsers.load_scan(dirpath)
    processed_scan = preprocess_scan(scan)
    return processed_scan


def process_zip(blob: storage.Blob, patient_id: str) -> None:
    os.mkdir('tmp')
    os.chdir('tmp')

    blob.download_to_filename(patient_id + '.zip')
    logging.info('extracting zip file')
    shutil.unpack_archive(patient_id + '.zip', format='zip')

    dirpath = list(os.walk('.'))[3][0]
    logging.info(f'loading scans from {dirpath}')
    scan = parsers.load_scan(dirpath)
    processed_scan = preprocess_scan(scan)
    logging.info(f'processing dicom data')

    os.chdir('..')
    shutil.rmtree('tmp')
    save_to_gcs(processed_scan, patient_id, blob.bucket)


def save_to_gcs(processed_scan, patient_id, bucket):
    outpath = f'numpy/{patient_id}.npy'
    stream = io.BytesIO()
    np.save(stream, processed_scan)
    processed_blob = storage.Blob(outpath, bucket=bucket)
    stream.seek(0)
    processed_blob.upload_from_file(stream)
    logging.info(f'saving dicom data to GCS in {outpath}')


def create_labels_csv(bucket, positives_df, negatives_df) -> None:
    """Creates a file labels.csv mapping patient_id to 1, if positive
    and 0, if negative.
    """
    labels = []
    blob: storage.Blob
    for blob in bucket.list_blobs(prefix=ELVOS_ANON + '/'):
        if blob.name.endswith('.csv'):
            continue  # Ignore the metadata CSV

        patient_id = blob.name[len(ELVOS_ANON) + 1: -EXTENSION_LENGTH]

        if patient_id in positives_df['Anon ID'].values:
            labels.append((patient_id, 1))
        elif patient_id in negatives_df['Anon ID'].values:
            labels.append((patient_id, 0))
        else:
            logging.warning(f'blob with patient id {patient_id} not found in'
                            f' the metadata CSVs, defaulting to negative')
            labels.append((patient_id, 0))
    labels_df = pd.DataFrame(labels, columns=['patient_id', 'label'])
    labels_df.to_csv('labels.csv', index=False)
    logging.info('uploading labels to the gs://elvos/labels.csv')
    labels_blob = storage.Blob('labels.csv', bucket=bucket)
    labels_blob.upload_from_filename('labels.csv')
    logging.info(f'label value counts {labels_df["label"].value_counts()}')


def preprocess_scan(slices: List[pydicom.FileDataset]) -> np.array:
    """Transforms the input dicom slices into a numpy array of pixels
    in Hounsfield units with standardized spacing.
    """
    scan = transforms.get_pixels_hu(slices)
    scan = transforms.standardize_spacing(scan, slices)
    return scan


def main():
    gcs_client = storage.Client(project='elvo-198322')
    input_bucket = gcs_client.get_bucket('elvos')

    positives_df, negatives_df = load_metadata(input_bucket)
    create_labels_csv(input_bucket, positives_df, negatives_df)

    blob: storage.Blob
    for blob in input_bucket.list_blobs(prefix=ELVOS_ANON + '/'):
        if blob.name.endswith('.csv'):
            continue  # Ignore the metadata CSV

        try:
            logging.info(f'processing blob {blob.name}')
            patient_id = blob.name[len(ELVOS_ANON) + 1: -EXTENSION_LENGTH]

            if blob.name.endswith('.cab'):
                process_cab(blob, patient_id)
            elif blob.name.endswith('.zip'):
                process_zip(blob, patient_id)
            else:
                logging.info(f'file extension must be .cab or .zip,'
                             f' got {blob.name}')
        except Exception as e:
            # Reset the working directory, tmp files for the next dataset
            os.chdir('..')
            shutil.rmtree('tmp')
            logging.error(e)


if __name__ == '__main__':
    configure_logger()
    main()
