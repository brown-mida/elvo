import logging
from blueno import io
from typing import Dict, Tuple

import keras
import numpy as np
import pandas as pd
from sklearn import model_selection

from blueno.types import ParamConfig, LukePipelineConfig, DataConfig


def clean_data(arrays: Dict[str, np.ndarray],
               labels: pd.DataFrame) -> \
        Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Handle duplicates in the dataframe and removes
    missing labels/arrays.

    The output dictionary and dataframe will have the same
    length.

    :param arrays:
    :param labels:
    :return:
    """
    filtered_arrays = arrays.copy()
    for patient_id in arrays:
        if patient_id not in labels.index.values:
            print(f'{patient_id} in arrays, but not in labels. Dropping')
            del filtered_arrays[patient_id]

    filtered_labels = labels.copy()
    print('Removing duplicate ids in labels:',
          filtered_labels[filtered_labels.index.duplicated()].index)
    filtered_labels = filtered_labels[~filtered_labels.index.duplicated()]

    for patient_id in filtered_labels.index.values:
        if patient_id not in arrays:
            print(f'{patient_id} in labels, but not in arrays. Dropping')
            filtered_labels = filtered_labels.drop(index=patient_id)

    assert len(filtered_arrays) == len(filtered_labels)
    return filtered_arrays, filtered_labels


def to_arrays(data: Dict[str, np.ndarray],
              labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the data and labels into numpy arrays.

    Note: This function filters mismatched labels.

    Note: The index of labels must be patient IDs.

    :param data:
    :param labels: a dataframe WITH patient ID for the index.
    :return: three arrays: the arrays, then the labels, then the corresponding
    ids
    """
    patient_ids = data.keys()
    X_list = []
    y_list = []
    remaining_ids = []
    for id_ in patient_ids:
        try:
            y_list += [labels.loc[id_]]
            X_list += [data[id_]]  # Needs to be in this order
            remaining_ids += [id_]
        except KeyError:
            logging.warning(f'{id_} in data was not present in labels')
            logging.warning(f'{len(X_list)}, {len(y_list)}')
    for id_ in labels.index.values:
        if id_ not in patient_ids:
            logging.warning(f'{id_} in labels was not present in data')

    assert len(X_list) == len(y_list)
    assert len(X_list) == len(remaining_ids)
    return np.stack(X_list), np.stack(y_list), np.array(remaining_ids)


def prepare_data(params: ParamConfig) -> Tuple[np.ndarray,
                                               np.ndarray,
                                               np.ndarray,
                                               np.ndarray,
                                               np.ndarray,
                                               np.ndarray,
                                               np.ndarray,
                                               np.ndarray]:
    """
    Prepares the data referenced in params for ML. This includes
    shuffling and expanding dims.

    :param params: a hyperparameter dictionary generated from a ParamGrid
    :return: x_train, x_valid, y_train, y_valid
    """
    logging.info(f'using params:\n{params}')
    # Load the arrays and labels
    data_params = params.data
    if isinstance(data_params, LukePipelineConfig):
        array_dict, labels_df = data_params.pipeline_callable(
            data_params.data_dir,
            data_params.labels_path,
            data_params.height_offset,
            data_params.mip_thickness,
            data_params.pixel_value_range,
        )
    elif isinstance(data_params, DataConfig):
        array_dict = io.load_arrays(data_params.data_dir)
        index_col = data_params.index_col
        labels_df = pd.read_csv(data_params.labels_path,
                                index_col=index_col)
    else:
        raise ValueError('params.data must be a subclass of DataConfig')

    label_col = data_params.label_col
    label_series = labels_df[label_col]

    # Convert to numpy arrays
    x, y, patient_ids = to_arrays(array_dict, label_series)

    if params.model.loss == keras.losses.categorical_crossentropy:
        y = keras.utils.to_categorical(y)
    elif params.model.loss == keras.losses.binary_crossentropy:
        if y.ndim == 1:
            y = np.expand_dims(y, axis=-1)

    assert y.ndim == 2

    logging.debug(f'x shape: {x.shape}')
    logging.debug(f'y shape: {y.shape}')
    logging.info(f'seeding to {params.seed} before shuffling')

    x_train, x_test, y_train, y_test, ids_train, ids_test = \
        model_selection.train_test_split(
            x, y, patient_ids,
            test_size=params.val_split,
            random_state=params.seed)

    x_train, x_valid, y_train, y_valid, ids_train, ids_valid = \
        model_selection.train_test_split(
            x_train, y_train, ids_train,
            test_size=params.val_split,
            random_state=params.seed)
    return (x_train, x_valid, x_test, y_train, y_valid, y_test,
            ids_train, ids_valid, ids_test)
