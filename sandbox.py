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


model = Resnet3DBuilder.build_resnet_18((64, 64, 64, 1), 1)
model.summary()
