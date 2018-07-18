import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from blueno import io, preprocessing, transforms

RAW_ARRAYS = {}


def preprocess_data(data_dir: str,
                    labels_dir: str,
                    height_offset: int,
                    mip_thickness: int,
                    pixel_value_range: Sequence,
                    test=False) -> Tuple[dict, pd.DataFrame]:
    """
    Preprocesses data by loading raw numpy data, cleaning, filtering,
    and transforming the arrays and labels.

    It is recommended that you use your own preprocessing callable
    if you want more configurability.

    :param data_dir: path to a directory containing .npz files
    :param labels_dir: path a directory containing the 2 metadata CSVs
    :param height_offset: the offset from the top of the 3D image to crop from
    :param mip_thickness: the thickness of a single mip
    :param pixel_value_range: the range of pixel values to crop to
    :return:
    """
    global RAW_ARRAYS
    if RAW_ARRAYS:
        logging.info('loading cached arrays')
        raw_arrays = RAW_ARRAYS
    else:
        logging.info('loading raw arrays from disk')
        if test:
            limit = 2  # Only load 2 npz files for speed
        else:
            limit = None
        raw_arrays = io.load_compressed_arrays(data_dir, limit=limit)
        RAW_ARRAYS = raw_arrays

    raw_labels = io.load_raw_labels(labels_dir)

    cleaned_arrays, cleaned_labels = preprocessing.clean_data(raw_arrays,
                                                              raw_labels)

    def filter_data(arrays: dict, labels: pd.DataFrame):
        filtered_arrays = {id_: arr for id_, arr in arrays.items()
                           if arr.shape[0] != 1}  # Remove the bad array
        filtered_labels = labels.loc[filtered_arrays.keys()]
        assert len(filtered_arrays) == len(filtered_labels)
        return filtered_arrays, filtered_labels

    filtered_arrays, filtered_labels = filter_data(cleaned_arrays,
                                                   cleaned_labels)

    def process_data(arrays: dict,
                     labels: pd.DataFrame,
                     height_offset: int,
                     mip_thickness: int,
                     pixel_value_range: Sequence):
        processed_arrays = {}
        for id_, arr in arrays.items():
            try:
                arr = transforms.crop(arr,
                                      (3 * mip_thickness, 200, 200),
                                      height_offset=height_offset)
                arr = np.stack([arr[:mip_thickness],
                                arr[mip_thickness: 2 * mip_thickness],
                                arr[2 * mip_thickness: 3 * mip_thickness]])
                arr = transforms.bound_pixels(arr,
                                              pixel_value_range[0],
                                              pixel_value_range[1])
                arr = arr.max(axis=1)
                arr = arr.transpose((1, 2, 0))
                processed_arrays[id_] = arr
            except AssertionError:
                print(f'patient id {id_} could not be processed,'
                      f' has input shape {arr.shape}')
        processed_labels = labels.loc[
            processed_arrays.keys()]  # Filter, if needed
        assert len(processed_arrays) == len(processed_labels)
        return processed_arrays, processed_labels

    processed_arrays, processed_labels = process_data(
        filtered_arrays,
        filtered_labels,
        height_offset=height_offset,
        mip_thickness=mip_thickness,
        pixel_value_range=pixel_value_range
    )
    return processed_arrays, processed_labels
