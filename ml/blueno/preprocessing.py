import typing

import numpy as np
import pandas as pd


def clean_data(arrays: typing.Dict[str, np.ndarray],
               labels: pd.DataFrame) -> \
        typing.Tuple[typing.Dict[str, np.ndarray], pd.DataFrame]:
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
