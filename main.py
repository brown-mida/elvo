import logging
import os
import time

import numpy as np
import pandas as pd
import scipy.ndimage
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from preprocessors import preprocessor, parsers, transforms

from generators.resnet_generator import ResnetGenerator
from generators.cad_generator import CadGenerator
from generators.augmented_generator import AugmentedGenerator

from models.resnet3d import Resnet3DBuilder
from models.autoencoder import Cad3dBuilder
from models.alexnet2d import AlexNet2DBuilder


def train_resnet():
    # Parameters
    dim_len = 192  # ~ 3 minutes per epoch
    top_len = 32
    epochs = 10
    batch_size = 16
    data_loc = '/home/shared/data/data-20180405'
    label_loc = '/home/shared/data/elvos_meta_drop1.xls'

    # Generators
    training_gen = AugmentedGenerator(data_loc, label_loc,
                                      dims=(dim_len, dim_len, top_len),
                                      batch_size=batch_size)
    validation_gen = AugmentedGenerator(data_loc, label_loc,
                                        dims=(dim_len, dim_len, top_len),
                                        batch_size=batch_size,
                                        validation=True)

    # Build and run model
    model = Resnet3DBuilder.build_resnet_34((dim_len, dim_len, top_len, 1), 1)
    model.compile(optimizer=Adam(lr=0.0001),
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
        verbose=1,
        max_queue_size=1)
    print('Model has been fit.')


def train_alexnet2d():
    # Parameters
    dim_len = 200
    top_len = 24
    epochs = 10
    batch_size = 32

    data_loc = '/home/shared/data/data-20180405'
    label_loc = '/home/shared/data/elvos_meta_drop1.xls'

    # Generators
    training_gen = AugmentedGenerator(data_loc, label_loc,
                                      dims=(dim_len, dim_len, top_len),
                                      batch_size=batch_size,
                                      extend_dims=False)
    validation_gen = AugmentedGenerator(data_loc, label_loc,
                                        dims=(dim_len, dim_len, top_len),
                                        batch_size=batch_size,
                                        extend_dims=False,
                                        validation=True)

    # Build and run model
    model = AlexNet2DBuilder.build((dim_len, dim_len, top_len))
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    mc_callback = ModelCheckpoint(filepath='tmp/alex_weights.hdf5', verbose=1)

    print('Model has been compiled.')
    model.fit_generator(
        generator=training_gen.generate(),
        steps_per_epoch=training_gen.get_steps_per_epoch(),
        validation_data=validation_gen.generate(),
        validation_steps=validation_gen.get_steps_per_epoch(),
        epochs=epochs,
        callbacks=[mc_callback],
        verbose=1,
        max_queue_size=1)
    print('Model has been fit.')


def train_cad():
    # Parameters
    epochs = 10
    batch_size = 1
    data_loc = '/home/shared/data/data-20180407'

    # Generators
    training_gen = CadGenerator(data_loc, batch_size=batch_size)
    validation_gen = CadGenerator(data_loc, batch_size=batch_size,
                                  validation=True)

    # Build and run model
    model = Cad3dBuilder.build((200, 200, 200,  1),
                               filters=(8, 8, 8))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])
    mc_callback = ModelCheckpoint(filepath='tmp/cad_weights.hdf5',
                                  verbose=1,
                                  save_weights_only=True)
    # tb_callback = TensorBoard(write_images=True)

    print('Model has been compiled.')
    model.fit_generator(
        generator=training_gen.generate(),
        steps_per_epoch=training_gen.get_steps_per_epoch(),
        validation_data=validation_gen.generate(),
        validation_steps=validation_gen.get_steps_per_epoch(),
        epochs=epochs,
        callbacks=[mc_callback],
        verbose=1,
        max_queue_size=1)
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
    # train_resnet()
    train_alexnet2d()
    # train_cad()
    end = int(time.time())
    # logging.debug('Preprocessing took {} seconds'.format(end - start))
