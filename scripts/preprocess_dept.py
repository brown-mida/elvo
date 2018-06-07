import logging
import os
import subprocess

import numpy as np


def crop(image3d: np.array):
    lw_center = len(image3d.shape[0]) / 2
    lw_min = lw_center - 100
    lw_max = lw_center + 100
    for i in range(image3d.shape[0]):
        if image3d[i].max() > 1000:
            height_min = i + 25
            height_max = height_min + 64
            return image3d[height_min:height_max, lw_min:lw_max, lw_min:lw_max]
    raise ValueError('Maximum pixel value is less than 1000. Could not crop.')


def standardize(image3d, min_bound=-1000, max_bound=400):
    image3d = (image3d - min_bound) / (max_bound - min_bound)
    image3d[image3d > 1] = 1.
    image3d[image3d < 0] = 0.
    return image3d


def split_numpy_data():
    """First 500 are training, remainder are validation"""
    filename_list = os.listdir('/home/lzhu7/data/numpy')
    training_list = filename_list[:500]
    validation_list = filename_list[500:]
    return training_list, validation_list


def preprocess():
    training, validation = split_numpy_data()
    for filename in training:
        try:
            logging.info('processing' + filename)
            subprocess.call(['scp',
                             'thingumy:/home/lzhu7/data/numpy/' + filename,
                             filename])
            image3d = np.load('/home/lzhu7/data/numpy/' + filename)
            image3d = crop(image3d)
            image3d = image3d.transpose((2, 1, 0))
            image3d = standardize(image3d)
            image3d.dump(filename)
            subprocess.call(['scp',
                             filename,
                             '/home/lzhu7/data/numpy_split/training' + filename])
            os.remove(filename)
        except Exception as e:
            logging.error(f'failed to process {filename} with error {e}')


if __name__ == '__main__':
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    preprocess()
