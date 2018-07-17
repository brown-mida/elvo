import pathlib
import typing
from typing import Dict

import keras
import numpy as np
import os

from blueno import utils


def load_arrays(data_dir: str) -> Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def load_model(model_path: str):
    # Need to do this otherwise the model won't load
    keras.metrics.sensitivity = utils.sensitivity
    keras.metrics.specificity = utils.specificity
    keras.metrics.true_positives = utils.true_positives
    keras.metrics.false_negatives = utils.false_negatives

    model = keras.models.load_model(model_path)
    return model


def load_compressed_arrays(data_dir: str) -> typing.Dict[str, np.ndarray]:
    data = dict()
    for filename in os.listdir(data_dir):
        print(f'Loading file {filename}')
        d = np.load(pathlib.Path(data_dir) / filename)
        data.update(d)  # merge all_data with d
    return data
