import pathlib
import typing

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt


def load_arrays(data_dir: str) -> typing.Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        print(f'Loading file {filename}')
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def load_compressed_arrays(data_dir: str) -> typing.Dict[str, np.ndarray]:
    data = dict()
    for filename in os.listdir(data_dir):
        print(f'Loading file {filename}')
        d = np.load(pathlib.Path(data_dir) / filename)
        data.update(d)  # merge all_data with d
    return data


def load_labels(labels_dir: str) -> pd.DataFrame:
    positives_df: pd.DataFrame = pd.read_csv(
        pathlib.Path(labels_dir) / 'positives.csv',
        index_col='Anon ID')
    positives_df['occlusion_exists'] = 1
    negatives_df: pd.DataFrame = pd.read_csv(
        pathlib.Path(labels_dir) / 'negatives.csv',
        index_col='Anon ID')
    negatives_df['occlusion_exists'] = 0
    return pd.concat([positives_df, negatives_df])


def clean_data(arrays: typing.Dict[str, np.ndarray],
               labels: pd.DataFrame) -> typing.Tuple[
    typing.Dict[str, np.ndarray], pd.DataFrame]:
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


def filter_data(arrays: typing.Dict[str, np.ndarray],
                labels: pd.DataFrame,
                variant: str):
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


def plot_images(data: typing.Dict[str, np.ndarray],
                labels: pd.DataFrame,
                num_cols=5,
                limit=20,
                offset=0):
    """
    Plots limit images in a single plot.

    :param data:
    :param labels:
    :param num_cols:
    :param limit: the number of images to plot
    :param offset:
    :return:
    """
    # Ceiling function of len(data) / num_cols
    num_rows = (min(len(data), limit) + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(10, 10))
    for i, patient_id in enumerate(data):
        if i < offset:
            continue
        if i >= offset + limit:
            break
        plot_num = i - offset + 1
        ax = fig.add_subplot(num_rows, num_cols, plot_num)
        ax.set_title(f'patient: {patient_id[:4]}...')
        label = ('positive' if labels.loc[patient_id]['occlusion_exists']
                 else 'negative')
        ax.set_xlabel(f'label: {label}')
        plt.imshow(data[patient_id])
    fig.tight_layout()
    plt.plot()


def crop(image3d: np.ndarray,
         output_shape: typing.Tuple[int, int, int],
         height_offset=30) -> np.ndarray:
    """
    Crops a 3d image in ijk form (height as axis 0).

    :param image3d:
    :param output_shape:
    :param height_offset:
    :return:
    """
    assert image3d.ndim == 3
    assert image3d.shape[1] == image3d.shape[2]
    assert output_shape[1] == output_shape[2]
    assert output_shape[1] <= image3d.shape[1]

    lw_center = image3d.shape[1] // 2
    lw_min = lw_center - output_shape[1] // 2
    lw_max = lw_center + output_shape[1] // 2
    for i in range(len(image3d) - 1, 0, -1):
        if image3d[i, lw_center, lw_center] >= 0:
            height_max = i - height_offset
            break
    else:
        raise ValueError('Failed to a relevant pixel'
                         ' with CT value of at least zero')
    height_min = height_max - output_shape[0]

    cropped = image3d[height_min:height_max, lw_min:lw_max, lw_min:lw_max]
    assert cropped.shape == output_shape
    return cropped


def crop_new(image3d: np.ndarray,
             output_shape,
             height_offset):
    pass


def bound_pixels(arr: np.ndarray,
                 min_bound: float,
                 max_bound: float) -> np.ndarray:
    arr[arr < min_bound] = min_bound
    arr[arr > max_bound] = max_bound
    return arr


def filter_pixels(arr: np.ndarray,
                  min_bound: float,
                  max_bound: float,
                  filter_value: float) -> np.ndarray:
    arr[arr < min_bound] = filter_value
    arr[arr > max_bound] = filter_value
    return arr


def average_intensity_projection():
    raise NotImplementedError()


def distance_intensity_projection():
    raise NotImplementedError()


def process_array(arr: np.ndarray,
                  preconfig: str):
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
                   (75, 20, 220),
                   height_offset=30)
        arr = np.stack([arr[:25], arr[25:50], arr[50:75]])
        arr = bound_pixels(arr, -40, 400)
        arr = arr.max(axis=1)
        arr = arr.transpose((1, 2, 0))
        assert arr.shape == (220, 220, 3)
        return arr
    # TODO: Optimize height offset, MIP thickness,
    # MIP overlap, (M)IP variations, bounding values, input shape
    # Save different configs as different if-statement blocks.

    raise ValueError(f'{preconfig} is not a valid preconfiguration')


def process_arrays(arrays: typing.Dict[str, np.ndarray],
                   preconfig: str) -> typing.Dict[str, np.ndarray]:
    processed = {}
    for id_, arr in arrays.items():
        try:
            processed[id_] = process_array(arr, preconfig)
        except AssertionError:
            print(f'patient id {id_} could not be processed,'
                  f' has input shape {arr.shape}')
    return processed


def process_data(arrays, labels, preconfig):
    print(f'Using config {preconfig}')
    processed_arrays = process_arrays(arrays, preconfig)
    processed_labels = labels.loc[processed_arrays.keys()]  # Filter, if needed
    assert len(processed_arrays) == len(processed_labels)
    return processed_arrays, processed_labels


def save_plots(arrays, labels, dirpath: str):
    os.mkdir(dirpath)
    num_plots = (len(arrays) + 19) // 20
    for i in range(num_plots):
        print(f'saving plot number {i}')
        plot_images(arrays, labels, 5, offset=20 * i)
        plt.savefig(f'{dirpath}/{20 * i}-{20 * i + 19}')


def save_data(arrays: typing.Dict[str, np.ndarray],
              labels: pd.DataFrame,
              dirpath: str,
              with_plots=True):
    os.makedirs(pathlib.Path(dirpath) / 'arrays')
    for id_, arr in arrays.items():
        # noinspection PyTypeChecker
        print(f'saving {id_}')
        np.save(pathlib.Path(dirpath) / 'arrays' / f'{id_}.npy', arr)
    labels.to_csv(pathlib.Path(dirpath) / 'labels.csv')
    plots_dir = str(pathlib.Path(dirpath) / 'plots')
    if with_plots:
        save_plots(arrays, labels, plots_dir)

# TODO: Uncomment before commit
# if __name__ == '__main__':
#     args = {
#         'arrays_dir': '/home/lzhu7/elvo-analysis/data/numpy_compressed/',
#         'labels_dir': '/home/lzhu7/elvo-analysis/data/metadata/',
#         'processed_dir': '/home/lzhu7/elvo-analysis/data/processed/',
#         'filter_variant': 'simple',
#         'process_variant': 'standard-crop-mip',
#     }
#     raw_arrays = load_compressed_arrays(args['arrays_dir'])
#     raw_labels = load_labels(args['labels_dir'])
#     cleaned_arrays, cleaned_labels = clean_data(raw_arrays, raw_labels)
#     filtered_arrays, filtered_labels = filter_data(cleaned_arrays,
#                                                    cleaned_labels,
#                                                    args['filter_variant'])
#     processed_arrays, processed_labels = process_data(filtered_arrays,
#                                                       filtered_labels,
#                                                       args['process_variant'])
#     save_data(processed_arrays, processed_labels, args['processed_dir'])
