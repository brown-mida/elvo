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
    """
    Upsample/transform a single array.

    :param arr: array to be upsampled
    :param file_id: file_id of that array
    :return:
    """
    # get all 3-length binary strings
    iterlist = list(itertools.product('01', repeat=3))
    # get all 3 planes to rotate upon
    axes = [[0, 1], [1, 2], [0, 2]]

    transform_number = 0

    # iterate through every plane
    for i in axes:
        # rotate
        rotated = np.rot90(arr, axes=i)
        # iterate through every binary string and perform all flips
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
            cloud.save_chunks_to_cloud(flipped, 'filtered',
                                       'positive', file_id_new)


def transform_positives():
    """
    Script that actually transforms and upsamples all the positives.

    :return:
    """
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')
    prefix = "chunk_data/filtered/positive"
    logging.info(f"transforming positive chunks from {prefix}")

    # for each blob in chunk_data/filtered/positive
    for in_blob in bucket.list_blobs(prefix=prefix):
        file_id = in_blob.name.split('/')[3]
        file_id = file_id.split('.')[0]

        # download chunk
        logging.info(f'downloading {in_blob.name}')
        input_arr = cloud.download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")

        # upsample chunk
        transform_one(input_arr, file_id)


def clean_old_data():
    """
    Removes old upsampled positives.

    :return:
    """
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')
    prefix = "chunk_data/normal/positive"

    # iterate through all blobs in the bucket
    for in_blob in bucket.list_blobs(prefix=prefix):
        logging.info(f'downloading {in_blob.name}')
        file_id = in_blob.name.split('/')[-1]
        file_id = file_id.split('.')[0]

        # delete it if it has an underscore in it
        if '_' in file_id:
            in_blob.delete()


def clean_new_data():
    """
    Removes non-upsampled positives

    :return:
    """
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')
    prefix = "chunk_data/filtered/positive"
    logging.info(f"transforming positive chunks from {prefix}")

    # iterate through all blobs in the bucket
    for in_blob in bucket.list_blobs(prefix=prefix):
        logging.info(f'downloading {in_blob.name}')
        file_id = in_blob.name.split('/')[-1]
        file_id = file_id.split('.')[0]

        # if there's no underscore, delete the blob
        if '_' in file_id:
            continue
        in_blob.delete()


def generate_csv():
    """
    Generates augmented annotated label csv file
    :return:
    """
    # Get old non-augmented labels
    labels_df = pd.read_csv('/home/harold_triedman/'
                            'elvo-analysis/annotated_labels.csv')

    # iterate through all rows
    for index, row in labels_df.iterrows():
        if row[1] == 1:
            # every time you come across a positive, add in 24 more rows
            to_add = {}
            for i in range(24):
                # we use index + 1 because filenames are 1-indexed
                new_patient_id = str(row[0]) + "_" + str(i + 1)
                to_add[index + 500000 + i] = [new_patient_id, 1]
            # add the augmented data to the end of the list
            to_add_df = pd.DataFrame.from_dict(
                to_add, orient='index', columns=['Unnamed: 0', 'label'])
            labels_df = labels_df.append(to_add_df)
            labels_df = labels_df.drop([index])
            print("Dropping patient " + str(index) + ": " + str(row[0]))

    labels_df.to_csv('augmented_annotated_labels.csv')


def clean_csv():
    """
    Takes out repeat positives (i.e. ones that have already been transformed
    but have not been removed from initial dataset)

    :return:
    """
    labels_df = pd.read_csv('/home/amy/data/augmented_annotated_labels.csv')
    print(len(labels_df))
    for index, row in labels_df.iterrows():
        if row[2] == 1 and '_' not in row[1]:
            labels_df = labels_df.drop(row[0])
    print(len(labels_df))
    labels_df = labels_df.drop(columns=['Unnamed: 0'])
    labels_df.to_csv('/home/amy/data/augmented_annotated_labels1.csv')


def run_transform():
    configure_logger()
    clean_csv()
    clean_old_data()
    generate_csv()
    transform_positives()
    clean_new_data()


if __name__ == '__main__':
    run_transform()
