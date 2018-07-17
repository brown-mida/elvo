import pathlib
import typing

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

import blueno.slack
from blueno.io import load_compressed_arrays, load_labels
from blueno.preprocessing import clean_data
from blueno.transforms import bound_pixels, crop


def filter_data(arrays: typing.Dict[str, np.ndarray],
                labels: pd.DataFrame,
                variant: str):
    # TODO(#64): Replace variant with parameters
    print(f'Using filter variant {variant}')
    if variant == 'simple':
        filtered_arrays = {id_: arr for id_, arr in arrays.items()
                           if arr.shape[0] != 1}  # Remove the bad array
        filtered_labels = labels.loc[filtered_arrays.keys()]
    elif variant == 'no-basvert':
        # Also remove array of height 1
        filtered_arrays = {id_: arr for id_, arr in arrays.items()
                           if arr.shape[0] != 1}  # Remove the bad array

        filtered_arrays = filtered_arrays.copy()
        col_name = 'Location of occlusions on CTA (Matt verified)'

        def condition(row):
            to_exclude = ('basilar', 'l vertebral', 'r vertebral',
                          'r vert', 'l vert')
            return (row[col_name] is np.nan
                    or row[col_name].lower().strip() not in to_exclude)

        filtered_ids = [id_ for id_ in list(filtered_arrays)
                        if condition(labels.loc[id_])]
        for id_ in list(filtered_arrays):
            if id_ not in filtered_ids:
                del filtered_arrays[id_]
        filtered_labels = labels.loc[filtered_ids]
    else:
        raise ValueError('Unsupported variant')
    assert len(filtered_arrays) == len(filtered_labels)
    return filtered_arrays, filtered_labels


def _process_array(arr: np.ndarray,
                   preconfig: str):
    # TODO(#64): Replace preconfig with parameters
    if preconfig == 'standard-crop-mip':
        arr = crop(arr,
                   (75, 200, 200),
                   height_offset=30)
        arr = np.stack([arr[:25], arr[25:50], arr[50:75]])
        arr = bound_pixels(arr, -40, 400)
        arr = arr.max(axis=1)
        arr = arr.transpose((1, 2, 0))
        assert arr.shape == (200, 200, 3)
        return arr
    if preconfig == '220-crop-mip':
        arr = crop(arr,
                   (75, 220, 220),
                   height_offset=30)
        arr = np.stack([arr[:25], arr[25:50], arr[50:75]])
        arr = bound_pixels(arr, -40, 400)
        arr = arr.max(axis=1)
        arr = arr.transpose((1, 2, 0))
        assert arr.shape == (220, 220, 3)
        return arr
    if preconfig == 'new-crop':
        arr = crop(arr,
                   (75, 220, 220),
                   height_offset=30)
        arr = np.stack([arr[:25], arr[25:50], arr[50:75]])
        arr = bound_pixels(arr, -40, 400)
        arr = arr.max(axis=1)
        arr = arr.transpose((1, 2, 0))
        assert arr.shape == (220, 220, 3)
        return arr

    if preconfig == 'lower-crop-mip':
        arr = crop(arr,
                   (75, 200, 200),
                   height_offset=40)
        arr = np.stack([arr[:25], arr[25:50], arr[50:75]])
        arr = bound_pixels(arr, -40, 400)
        arr = arr.max(axis=1)
        arr = arr.transpose((1, 2, 0))
        assert arr.shape == (200, 200, 3)
        return arr

    # TODO(#65): Optimize height offset, MIP thickness,
    # MIP overlap, (M)IP variations, bounding values, input shape
    # Save different configs as different if-statement blocks.

    raise ValueError(f'{preconfig} is not a valid preconfiguration')


def _process_arrays(arrays: typing.Dict[str, np.ndarray],
                    preconfig: str) -> typing.Dict[str, np.ndarray]:
    processed = {}
    for id_, arr in arrays.items():
        try:
            processed[id_] = _process_array(arr, preconfig)
        except AssertionError:
            print(f'patient id {id_} could not be processed,'
                  f' has input shape {arr.shape}')
    return processed


def process_data(arrays, labels, preconfig):
    print(f'Using config {preconfig}')
    processed_arrays = _process_arrays(arrays, preconfig)
    processed_labels = labels.loc[processed_arrays.keys()]  # Filter, if needed
    assert len(processed_arrays) == len(processed_labels)
    return processed_arrays, processed_labels


def save_plots(arrays, labels, dirpath: str):
    os.mkdir(dirpath)
    num_plots = (len(arrays) + 19) // 20
    for i in range(num_plots):
        print(f'saving plot number {i}')
        blueno.slack.plot_images(arrays, labels, 5, offset=20 * i)
        plt.savefig(f'{dirpath}/{20 * i}-{20 * i + 19}')


def save_data(arrays: typing.Dict[str, np.ndarray],
              labels: pd.DataFrame,
              dirpath: str,
              with_plots=True):
    """
    Saves the arrays and labels in the given dirpath.

    :param arrays:
    :param labels:
    :param dirpath:
    :param with_plots:
    :return:
    """
    # noinspection PyTypeChecker
    os.makedirs(pathlib.Path(dirpath) / 'arrays')
    for id_, arr in arrays.items():
        print(f'saving {id_}')
        # noinspection PyTypeChecker
        np.save(pathlib.Path(dirpath) / 'arrays' / f'{id_}.npy', arr)
    labels.to_csv(pathlib.Path(dirpath) / 'labels.csv')
    plots_dir = str(pathlib.Path(dirpath) / 'plots')
    if with_plots:
        save_plots(arrays, labels, plots_dir)


def bluenop(args: dict):
    """
    Runs a preprocessing job with the args.

    :param args: The config dictionary for the processing job
    :return:
    """
    raw_arrays = load_compressed_arrays(args['arrays_dir'])
    raw_labels = load_labels(args['labels_dir'])
    cleaned_arrays, cleaned_labels = clean_data(raw_arrays, raw_labels)
    filtered_arrays, filtered_labels = filter_data(cleaned_arrays,
                                                   cleaned_labels,
                                                   args['filter_variant'])
    processed_arrays, processed_labels = process_data(filtered_arrays,
                                                      filtered_labels,
                                                      args['process_variant'])
    save_data(processed_arrays, processed_labels, args['processed_dir'])

# if __name__ == '__main__':
#     arguments = {
#         'arrays_dir': '/home/lzhu7/elvo-analysis/data/numpy_compressed/',
#         'labels_dir': '/home/lzhu7/elvo-analysis/data/metadata/',
#         'processed_dir': '/home/lzhu7/elvo-analysis/data/processed/',
#         'filter_variant': 'simple',
#         'process_variant': 'standard-crop-mip',
#     }
#     bluenop(arguments)
