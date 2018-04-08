import logging
import os
import time

import numpy as np
import pandas as pd
import scipy.ndimage
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from preprocessors import preprocessor, parsers, transforms
from generators.t_generator import Generator
from generators.cad_generator import CadGenerator

from models.resnet3d import Resnet3DBuilder
from models.autoencoder import Cad3dBuilder

data_loc = '/home/shared/data/data-20180407'
gen = CadGenerator(data_loc, batch_size=16)

while True:
    a = gen.generate()
    a = next(a)
    print(a)
    print(np.shape(a))
