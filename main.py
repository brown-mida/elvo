import logging
import os
import time

import numpy as np
import pandas as pd
import scipy.ndimage
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from preprocessors import preprocessor, parsers, transforms

def load_processed_data(dirpath):
    # Reading in the data
    patient_ids = []
    images = []
    for i, filename in enumerate(os.listdir(dirpath)):
        if 'csv' in filename:
            continue
        if i > 160:
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
    for _, row in labels.sample(frac=1).iterrows():
        for i, id_ in enumerate(patient_ids):
            if row['PatientID'] == id_:
                y[i] = (row['ELVO status'] == 'Yes')
    print('Parsed labels')
    print('Transformed data')
    return X, y


def summarize_labels(y):
    print('Number of positives', sum(y == 1))
    print('Number of validation positives', sum(y[-40:0] == 1))
    return

def train_resnet():
    from models.resnet3d import Resnet3DBuilder
    dim_length = 64  # ~ 3 minutes per epoch
    epochs = 10
    X, y = load_and_transform('data-1521428185', dim_length)
    summarize_labels(y)

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
              batch_size=16,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[mc_callback],
              verbose=2)
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
