import logging
import os
import time

import numpy as np
import pandas as pd
import scipy.ndimage
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from preprocessors import preprocessor, parsers, transforms
from generator import Generator

from models.resnet3d import Resnet3DBuilder


def train_resnet():
    # Parameters
    dim_length = 64  # ~ 3 minutes per epoch
    epochs = 10
    batch_size = 16
    data_loc = 'data-1521428185'
    label_loc = '/home/lukezhu/data/ELVOS/elvos_meta_drop1.xls'

    # Generators
    training_gen = Generator(data_loc, label_loc, dim_length=dim_length, 
                             batch_size=batch_size)
    validation_gen = Generator(data_loc, label_loc, dim_length=dim_length, 
                               batch_size=batch_size, validation=True)

    # Build and run model
    model = Resnet3DBuilder.build_resnet_18((dim_length, dim_length,
                                             dim_length, 1), 1)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    mc_callback = ModelCheckpoint(filepath='tmp/weights.hdf5', verbose=1)
    # tb_callback = TensorBoard(write_images=True)
    
    print('Model has been compiled.')
    model.fit_generator(
        generator=training_gen.generate(),
        steps_per_epoch=training_gen.get_steps_per_epoch(),
        validation_data=validation_gen.generate(),
        validation_steps=validation_gen.get_steps_per_epoch(),
        epochs=epochs,
        callbacks=[mc_callback],
        verbose=2)
    print('Model has been fit.')


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
    # logging.debug('Preprocessing took {} seconds'.format(end - start))
