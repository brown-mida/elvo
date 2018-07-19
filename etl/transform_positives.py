"""
Purpose: This script takes all the positive ROI-cropped 32x32x32 "chunks,"
transforms them via vertical flips and rotations, and adds them to the GCS
storage folder "positives + augmentation."
"""

import itertools
import logging

import numpy as np
import pandas as pd

# from matplotlib import pyplot as plt
from lib import cloud_management as cloud


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def transform_one(arr, file_id):
    iterlist = list(itertools.product('01', repeat=3))
    axes = [[0, 1], [1, 2], [0, 2]]

    transform_number = 0
    for i in axes:
        rotated = np.rot90(arr, axes=i)
        for j in iterlist:
            transform_number += 1
            flipped = arr
            if j[0] == '1':
                flipped = np.flipud(rotated)
            if j[1] == '1':
                flipped = np.fliplr(rotated)
            if j[2] == '1':
                flipped = rotated[:, :, ::-1]
                # save to the numpy generator source directory
            file_id_new = file_id + "_" + str(transform_number)
            cloud.save_chunks_to_cloud(flipped, 'normal',
                                       'positive', file_id_new)


def transform_positives():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # iterate through every source directory...
    prefix = "chunk_data/normal/positive"
    logging.info(f"transforming positive chunks from {prefix}")

    for in_blob in bucket.list_blobs(prefix=prefix):
        # blacklist
        if in_blob.name == prefix + 'LAUIHISOEZIM5ILF.npy':
            continue

        # perform the normal cropping procedure
        logging.info(f'downloading {in_blob.name}')
        file_id = in_blob.name.split('/')[3]
        file_id = file_id.split('.')[0]

        input_arr = cloud.download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")
        # crop individual input array
        transform_one(input_arr, file_id)

        # plt.figure(figsize=(6, 6))
        # plt.imshow(not_extreme_arr, interpolation='none')
        # plt.show()


def clean_old_data():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # iterate through every source directory...
    prefix = "chunk_data/normal/positive"
    logging.info(f"transforming positive chunks from {prefix}")

    i = 0
    for in_blob in bucket.list_blobs(prefix=prefix):
        i += 1
        # blacklist
        if in_blob.name == prefix + 'LAUIHISOEZIM5ILF.npy':
            continue

        # perform the normal cropping procedure
        logging.info(f'downloading {in_blob.name}')
        file_id = in_blob.name.split('/')[-1]
        file_id = file_id.split('.')[0]

        if '_' in file_id:
            in_blob.delete()

    logging.info(i)


def clean_new_data():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # iterate through every source directory...
    prefix = "chunk_data/normal/positive"
    logging.info(f"transforming positive chunks from {prefix}")

    i = 0
    for in_blob in bucket.list_blobs(prefix=prefix):
        i += 1
        # blacklist
        if in_blob.name == prefix + 'LAUIHISOEZIM5ILF.npy':
            continue

        # perform the normal cropping procedure
        logging.info(f'downloading {in_blob.name}')
        file_id = in_blob.name.split('/')[-1]
        file_id = file_id.split('.')[0]

        if '_' in file_id:
            continue

        in_blob.delete()
    logging.info(i)


def generate_csv():
    labels_df = pd.read_csv('/home/harold_triedman/'
                            'elvo-analysis/annotated_labels.csv')
    for index, row in labels_df.iterrows():
        logging.info(index, row[1])
        if row[1] == 1:
            # every time you come across a positive, add in 24 more rows
            to_add = {}
            for i in range(24):
                # we use index + 1 because filenames are 1-indexed
                new_patient_id = str(row[0]) + "_" + str(i + 1)
                to_add[index + 500000 + i] = [new_patient_id, 1]
            to_add_df = pd.DataFrame.from_dict(
                to_add, orient='index', columns=['Unnamed: 0', 'label'])
            logging.info(to_add_df)
            labels_df = labels_df.append(to_add_df)
    labels_df.to_csv("augmented_annotated_labels.csv")


def run_transform():
    configure_logger()
    clean_old_data()
    generate_csv()
    transform_positives()
    clean_new_data()


if __name__ == '__main__':
    run_transform()
