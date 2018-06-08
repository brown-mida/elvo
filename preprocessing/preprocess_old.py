import logging
import os
import shutil

import numpy as np
import pandas as pd
from google.cloud import storage

import preprocessing.parsers as parsers
from preprocessing.preprocess import preprocess_scan


def preprocess(bucket_name: str, roi_dir: str, output_dir: str):
    """Loads the data from the input directory and saves
    normalized, zero-centered, 200 x 200 x 200 3D renderings
    in output_dir.
    """
    # TODO: Remove hardcoded path
    df = pd.read_excel('/home/shared/data/elvos_meta_drop1.xls')

    os.makedirs(output_dir)
    for id_ in df['PatientID']:
        try:
            logging.info('Preprocessing scans for patient {}'.format(id_))
            filename = id_ + '.zip'
            blob_path = 'ELVOs_anon/{}'.format(filename)
            _download_blob(bucket_name, blob_path, filename)
            logging.info('Downloaded the data')
            shutil.unpack_archive(filename, format='zip')
            logging.info('Unzipped the data')
            # For some reason, we need to go two levels deep
            scans_path_root = [
                path for path in os.listdir('.')
                if path.startswith(id_) and not path.endswith('.zip')
            ][0]
            scans_path = scans_path_root
            scans_path += '/' + os.listdir(scans_path_root)[0]
            scans_path += '/' + os.listdir(scans_path)[0]
            slices = parsers.load_scan(scans_path)
            logging.info('Loaded slices into memory')
            scan = preprocess_scan(slices)
            logging.info(
                'Finished converting to HU, standardizing pixels'
                ' to 1mm, and cropping the array to 200x200x200'
            )
            _save_scan(id_, scan, output_dir)
            logging.info('Finished saving scan as a .npy file')
            os.remove(filename)
            shutil.rmtree(scans_path_root)
            logging.info('Removed scan from local filesystem')
        except Exception:
            # TODO(Luke): Remove after first run
            logging.exception(
                'Something failed while processing the scans for'
                ' patient {}'.format(id_)
            )

    # Consider doing this step just before training the model
    # normalized = normalize(np.stack(processed_scans))
    _save_info(df['PatientID'], roi_dir, output_dir)


def _download_blob(bucket_name: str,
                   source_blob_name: str,
                   destination_file_name: str):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)


def _save_scan(id_, scan, output_dir):
    """Saves the scan for patient id_ in the output_dir
    as a numpy file."""
    outfile = '{}/patient-{}'.format(output_dir, id_)
    np.save(outfile, scan)


def _save_info(patient_ids, roi_dir, output_dir):
    """Saves labels by matching directory names in roi_dir to
    patient ids. Also saves the bounding boxes.
    """
    columns = [
        'label',
        'centerX', 'centerY', 'centerZ',
        'deviationX', 'deviationY', 'deviationZ'
    ]
    info = pd.DataFrame(index=patient_ids, columns=columns)
    info['label'] = 0  # Set a default

    for name in os.listdir(roi_dir):
        try:
            path = '{}/{}/AnnotationROI.acsv'.format(roi_dir, name)
            center, dev = parsers.parse_bounding_box(path)
            info.loc[name, 'label'] = 1
            info.loc[name, ['centerX', 'centerY', 'centerZ']] = center
            info.loc[name, ['deviationX', 'deviationY', 'deviationZ']] = dev
        except OSError:
            pass

    info.to_csv('{}/labels.csv'.format(output_dir))
