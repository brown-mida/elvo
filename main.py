import os
import time

import numpy as np
import pandas as pd

import parsers
import transforms


def preprocess(input_dir, output_dir):
    """Loads the data from the input directory and saves
    normalized, zero-centered, 200 x 200 x 200 3D renderings
    in output_dir.
    """
    # Input (NOTE: Keep I/O separate from transformations for clarity!)
    patient_ids, preprocessed_scans = parsers.load_scans(input_dir)

    # Transformations
    processed_scans = []
    for id_, slices in zip(patient_ids, preprocessed_scans):
        scan = transforms.get_pixels_hu(slices)
        scan = transforms.standardize_spacing(scan, slices)
        scan = transforms.crop(scan)
        # TODO: Generate an image at this point to verify the preprocessing
        print('Finished converting to HU, standardizing pixels'
              ' to 1mm, and cropping the array to 200x200x200')
        processed_scans.append(scan)

    # Consider doing this step just before training the model
    # normalized = normalize(np.stack(processed_scans))
    # Output
    # TODO: Consider moving this step into a separate function
    os.makedirs(output_dir)
    for id_, scan in zip(patient_ids, processed_scans):
        outfile = '{}/patient-{}'.format(output_dir, id_)
        np.save(outfile, scan)

    columns = [
        'label',
        'centerX', 'centerY', 'centerZ',
        'deviationX', 'deviationY', 'deviationZ'
    ]
    labels = pd.DataFrame(index=patient_ids, columns=columns)
    labels['label'] = 0  # Set a default

    for name in os.listdir(input_dir):
        if name.isdigit():
            path = '{}/{}/AnnotationROI.acsv'.format(input_dir, name)
            center, dev = parsers.parse_bounding_box(path)
            labels.loc[name, 'label'] = 1
            labels.loc[name, ['centerX', 'centerY', 'centerZ']] = center
            labels.loc[name, ['deviationX', 'deviationY', 'deviationZ']] = dev

    labels.to_csv('{}/labels.csv'.format(output_dir))


if __name__ == '__main__':
    IN_DIR = 'RI Hospital ELVO Data'  # The relative path to the dataset
    OUT_DIR = 'data-{}'.format(int(time.time()))
    preprocess(IN_DIR, OUT_DIR)
