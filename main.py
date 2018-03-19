import logging
import os
import time

import numpy as np
import pandas as pd
import scipy.ndimage
from keras.callbacks import TensorBoard, ModelCheckpoint

import parsers
import transforms


def preprocess(ct_dir, roi_dir, output_dir):
    """Loads the data from the input directory and saves
    normalized, zero-centered, 200 x 200 x 200 3D renderings
    in output_dir.
    """
    parsers.unzip_scans(ct_dir)
    logging.debug('Unzipped data in {}'.format(ct_dir))
    patient_ids = parsers.load_patient_infos(ct_dir)
    logging.debug('Loaded patient ids in {}'.format(ct_dir))

    os.makedirs(output_dir)
    for id_, path in patient_ids.items():
        try:
            slices = parsers.load_scan(path)
            logging.debug('Loaded slices for patient {}'.format(id_))
            scan = _preprocess_scan(slices)
            _save_scan(id_, scan, output_dir)
        except Exception as e:
            # TODO(Luke): Remove after first run
            logging.error('Failed to load {}'.format(id_))
            logging.error(e)

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


def load_processed_data(dirpath):
    # Reading in the data
    patient_ids = []
    images = []
    for i, filename in enumerate(os.listdir(dirpath)):
        if 'csv' in filename:
            continue
        if i > 200:
            break
        patient_ids.append(filename[8:-4])
        images.append(np.load(dirpath + '/' + filename))
        print('Loading image {}'.format(i))
    return images, patient_ids

def transform_images(images, dim_length):
    resized = np.stack([scipy.ndimage.interpolation.zoom(arr, dim_length / 200)
                    for arr in images])
    print('Resized data')
    normalized = transforms.normalize(resized)
    print('Normalized data')
    return np.expand_dims(normalized, axis=4)


def load_and_transform(dirpath, dim_length):
    images, patient_ids = load_processed_data(dirpath)
    labels = pd.read_excel('/home/lukezhu/data/ELVOS/elvos_meta_drop1.xls')
    print('Loaded data')

    X = transform_images(images, dim_length)
    y = np.zeros(len(patient_ids))
    for _, row in labels.iterrows():
        for i, id_ in enumerate(patient_ids):
            if row['PatientID'] == id_:
                y[i] = (row['ELVO status'] == 'Yes')
    print('Parsed labels')
    print('Transformed data')
    return X, y


def train_resnet():
    from models.resnet3d import Resnet3DBuilder
    dim_length = 32  # ~ 3 minutes per epoch
    epochs = 10
    X, y = load_and_transform('data-1521428185', dim_length)
    model = Resnet3DBuilder.build_resnet_18((dim_length, dim_length,
                                             dim_length, 1),
                                            1)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    mc_callback = ModelCheckpoint(filepath='tmp/weights.hdf5', verbose=1)
    # tb_callback = TensorBoard(write_images=True)
    print('Compiled model')
    model.fit(X, y,
              epochs=epochs, validation_split=0.2,
              callbacks=[mc_callback], verbose=2)
    print('Fit model')


if __name__ == '__main__':
    # TODO (Make separate loggers)
    # TODO (Split this file into separate scripts)
    logging.basicConfig(filename='logs/preprocessing.log', level=logging.DEBUG)
    start = int(time.time())
    OUTPUT_DIR = 'data-{}'.format(start)
    logging.debug('Saving processed data to {}'.format(OUTPUT_DIR))
    # preprocess('RI Hospital ELVO Data', 'RI Hospital ELVO Data', OUTPUT_DIR)
    # preprocess('ELVOS/anon', 'ELVOS/ROI_cropped', OUTPUT_DIR)
    # preprocess('../data/ELVOS/anon', '../data/ELVOS/ROI_cropped', OUTPUT_DIR)
    train_resnet()
    end = int(time.time())
    logging.debug('Preprocessing took {} seconds'.format(end - start))
