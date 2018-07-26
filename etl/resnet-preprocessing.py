"""
This is a script adapted from Andrew's Resnet Preprocessing.ipynb.
This is a more refined version of Mary's primitive skull-stripping data,
used to produce numpy arrays used as training data on the Resnet-50 model.
"""

import os
import typing
import logging

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# first time: run these commands to get raw data and labels
# !gsutil -m rsync -r gs://elvos/airflow/npy /home/mdong/elvo-analysis/
# data/skull-stripped-resnet/npy

# !gsutil cp gs://elvos/labels.csv /home/mdong/elvo-analysis/data/
# skull-stripped-resnet/labels.csv


data_path = '/home/mdong/elvo-analysis/data/skull-stripped-resnet/'


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def load_data(data_dir: str) -> typing.Dict[str, np.ndarray]:
    """Returns a dictionary which maps patient ids
    to patient pixel data."""
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(data_dir + '/' + filename)
        print(f'loading {filename} into data_dict')
    return data_dict


# Preprocessing Part I: Remove bad data and duplicate labels
def process_images(data: typing.Dict[str, np.ndarray]):
    return {id_: arr for id_, arr in data.items() if
            len(arr) != 1}  # Remove the bad image


def process_labels(labels: pd.DataFrame, data: typing.Dict[str, np.ndarray]):
    # TODO: Remove duplicate HLXOSVDF27JWNCMJ, IYDXJTFVWJEX36DO from ELVO_key
    labels = labels.loc[~labels.index.duplicated()]  # Remove duplicate ids
    labels = labels.loc[list(data.keys())]
    assert len(labels) == len(data)
    return labels


def plot_images(data: typing.Dict[str, np.ndarray],
                labels: pd.DataFrame,
                num_cols: int,
                limit=20,
                offset=0):
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
        label = 'positive' if labels.loc[patient_id]["label"] else 'negative'
        ax.set_xlabel(f'label: {label}')
        plt.imshow(data[patient_id])
    fig.tight_layout()
    plt.plot()


if __name__ == '__main__':
    # define labels and images
    labels_df = pd.read_csv(data_path + '/labels.csv',
                            index_col='patient_id')
    data_dict = load_data(data_path + 'npy')
    data_dict = process_images(data_dict)
    labels_df = process_labels(labels_df, data_dict)
    plot_images({k: arr for k, arr in data_dict.items()}, labels_df, 5,
                offset=20)
