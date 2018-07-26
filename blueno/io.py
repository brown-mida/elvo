import pathlib
import typing
from typing import Dict

import keras
import numpy as np
import os
import pandas as pd

from blueno import utils


def load_arrays(data_dir: str) -> Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def load_model(model_path: str, compile=True):
    # Need to do this otherwise the model won't load
    keras.metrics.sensitivity = utils.sensitivity
    keras.metrics.specificity = utils.specificity
    keras.metrics.true_positives = utils.true_positives
    keras.metrics.false_negatives = utils.false_negatives

    model = keras.models.load_model(model_path, compile=compile)
    return model


def load_compressed_arrays(data_dir: str,
                           limit=None) -> typing.Dict[str, np.ndarray]:
    """Loads a directory containing npz files.

    The keys will be the keys of the loaded npz dict.
    """
    data = dict()
    filenames = os.listdir(data_dir)
    if limit:
        filenames = filenames[:limit]
    for filename in filenames:
        print(f'Loading file {filename}')
        d = np.load(pathlib.Path(data_dir) / filename)
        data.update(d)  # merge all_data with d
    return data


def load_raw_labels(labels_dir: str, index_col='Anon ID') -> pd.DataFrame:
    """Loads a directory containing a postives.csv and negatives.csv
    file."""
    positives_df: pd.DataFrame = pd.read_csv(
        pathlib.Path(labels_dir) / 'positives.csv',
        index_col=index_col)
    positives_df['occlusion_exists'] = 1
    negatives_df: pd.DataFrame = pd.read_csv(
        pathlib.Path(labels_dir) / 'negatives.csv',
        index_col=index_col)
    negatives_df['occlusion_exists'] = 0
    return pd.concat([positives_df, negatives_df])
