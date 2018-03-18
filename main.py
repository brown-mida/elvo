import logging
import os
import time

import numpy as np
import pandas as pd

import parsers
import transforms


def preprocess(ct_dir, roi_dir, output_dir):
    """Loads the data from the input directory and saves
    normalized, zero-centered, 200 x 200 x 200 3D renderings
    in output_dir.
    """
    parsers.unzip_scans(ct_dir)
    patient_ids = parsers.load_patient_infos(ct_dir)
    logging.debug('Loaded patient ids in {}'.format(ct_dir))

    os.makedirs(output_dir)

    for id_, path in patient_ids.items():
        slices = parsers.load_scan(path)
        logging.debug('Loaded slices for patient {}'.format(id_))
        scan = _preprocess_scan(slices)
        _save_scan(id_, scan, output_dir)

    # Consider doing this step just before training the model
    # normalized = normalize(np.stack(processed_scans))

    _save_info(patient_ids, roi_dir, output_dir)


def _preprocess_scan(slices):
    """Transforms the CT slices into a processed 3D numpy array.
    """
    scan = transforms.get_pixels_hu(slices)
    scan = transforms.standardize_spacing(scan, slices)
    scan = transforms.crop(scan)
    # TODO: Generate an image at this point to verify the preprocessing
    logging.debug(
        'Finished converting to HU, standardizing pixels'
        ' to 1mm, and cropping the array to 200x200x200'
    )
    return scan


def _save_scan(id_, scan, output_dir):
    """Saves the scan for patient id_ in the output_dir
    as a numpy file."""
    outfile = '{}/patient-{}'.format(output_dir, id_)
    np.save(outfile, scan)
    logging.debug(
        'Finished saving scan as a .npy file'
    )


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


if __name__ == '__main__':
    logging.basicConfig(filename='preprocessing.log', level=logging.DEBUG)
    start = int(time.time())
    OUTPUT_DIR = 'data-{}'.format(start)
    logging.debug('Saving processed data to {}'.format(OUTPUT_DIR))
    # preprocess('RI Hospital ELVO Data', 'RI Hospital ELVO Data', OUT_DIR)
    preprocess('ELVOS/anon', 'ELVOS/ROI_cropped', OUTPUT_DIR)
    end = int(time.time())
    logging.debug('Preprocessing took {} seconds'.format(end - start))
