"""
A script to do preprocessing for the ROI-prediction-based classifier model.
"""
import csv
import numpy as np
import random
from google.cloud import storage
from lib import cloud_management
import pickle
import logging
import pandas as pd


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def main():
    configure_logger()
    # Access Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'
        # 'credentials/client_secret.json'
    )
    bucket = gcs_client.get_bucket('elvos')

    # Get label data from Google Cloud Storage
    blob = storage.Blob('augmented_annotated_labels.csv', bucket)
    blob.download_to_filename('tmp/augmented_annotated_labels.csv')
    prelim_label_data = {}

    # load labels from augmented_annotated_labels.csv
    with open('tmp/augmented_annotated_labels.csv', 'r') as pos_file:
        reader = csv.reader(pos_file, delimiter=',')
        for row in reader:
            if row[1] != 'Unnamed: 0.1':
                prelim_label_data[row[1]] = int(row[2])
                # prelim_label_data[row[2]] = int(row[3])

    # Get all of the positives from the label data
    positive_label_data = {}
    logging.info('getting 12168 positive labels')
    for id_, label in list(prelim_label_data.items()):
        if label == 1 and '_' in id_:
            positive_label_data[id_] = label

    positive_train_label_data = {}
    positive_val_label_data = {}
    train = {}
    val = {}

    # Loop through positives
    for i, id_ in enumerate(list(positive_label_data.keys())):
        if i % 24 == 0:

            # Split into train/val sets based off of random flips
            seed = random.randint(1, 100)
            stripped_id = id_[:-1]
            meta_id = id_[:16]

            # add ID to positive train data and train metadata
            if seed > 10:
                positive_train_label_data[id_] = 1
                train[meta_id] = ''
                for j in range(2, 25):
                    positive_train_label_data[stripped_id + str(j)] = 1

            # add ID to positive val data and val metadata
            else:
                positive_val_label_data[id_] = 1
                val[meta_id] = ''
                for j in range(2, 25):
                    positive_val_label_data[stripped_id + str(j)] = 1

    # Get 14500 random negatives from the label data to feed into our generator
    negative_counter = 0
    negative_train_label_data = {}
    negative_val_label_data = {}
    logging.info("getting 14500 random negative labels")
    while negative_counter < 14500:

        # Get random chunk
        id_, label = random.choice(list(prelim_label_data.items()))

        # if it's a negative
        if label == 0:
            if negative_counter % 500 == 0:
                logging.info(f'gotten {negative_counter} labels so far')

            meta_id = id_[:16]
            # if another chunk in this brain is in train metadata dict
            if meta_id in train:
                negative_train_label_data[id_] = label

            # else if another chunk in this brain is in val metadata dict
            elif meta_id in val:
                negative_val_label_data[id_] = label

            # otherwise flip a coin to see where it's going to end up
            else:
                seed = random.randint(1, 100)
                if seed > 10:
                    negative_train_label_data[id_] = label
                    train[meta_id] = ''
                else:
                    negative_val_label_data[id_] = label
                    val[meta_id] = ''

            # delete it from prelim_label_data to ensure no re-picks
            del prelim_label_data[id_]
            negative_counter += 1

    # save train/val metadata
    train_df = pd.DataFrame.from_dict(train, orient='index')
    val_df = pd.DataFrame.from_dict(val, orient='index')
    train_df.to_csv('train_ids.csv')
    val_df.to_csv('val_ids.csv')

    train_chunks = []
    train_labels = []
    val_chunks = []
    val_labels = []

    # Get positive train chunks
    i = 1
    for id_, label in list(positive_train_label_data.items()):
        if i % 500 == 0:
            logging.info(f'got chunk {i}')
        i += 1
        blob = bucket.get_blob('chunk_data/normal/positive/' + id_ + '.npy')
        arr = cloud_management.download_array(blob)
        if arr.shape == (32, 32, 32):
            arr = np.expand_dims(arr, axis=-1)
            train_chunks.append(arr)
            train_labels.append(label)
    logging.info(f'{i} total positive training chunks')

    # Get positive val chunks
    i = 1
    for id_, label in list(positive_val_label_data.items()):
        if i % 500 == 0:
            logging.info(f'got chunk {i}')
        i += 1
        blob = bucket.get_blob('chunk_data/normal/positive/' + id_ + '.npy')
        arr = cloud_management.download_array(blob)
        if arr.shape == (32, 32, 32):
            arr = np.expand_dims(arr, axis=-1)
            val_chunks.append(arr)
            val_labels.append(label)
    logging.info(f'{i} total positive validation chunks')

    # Get negative train chunks
    i = 1
    for id_, label in list(negative_train_label_data.items()):
        if i % 500 == 0:
            logging.info(f'got chunk {i}')
        i += 1
        blob = bucket.get_blob('chunk_data/normal/negative/' + id_ + '.npy')
        arr = cloud_management.download_array(blob)
        if arr.shape == (32, 32, 32):
            arr = np.expand_dims(arr, axis=-1)
            train_chunks.append(arr)
            train_labels.append(label)
    logging.info(f'{i} total negative chunks')

    # Get negative val chunks
    i = 1
    for id_, label in list(negative_val_label_data.items()):
        if i % 500 == 0:
            logging.info(f'got chunk {i}')
        i += 1
        blob = bucket.get_blob('chunk_data/normal/negative/' + id_ + '.npy')
        arr = cloud_management.download_array(blob)
        if arr.shape == (32, 32, 32):
            arr = np.expand_dims(arr, axis=-1)
            val_chunks.append(arr)
            val_labels.append(label)
    logging.info(f'{i} total negative chunks')

    # shuffle order of training data
    tmp = list(zip(train_chunks, train_labels))
    random.shuffle(tmp)
    train_chunks, train_labels = zip(*tmp)

    # shuffle order of validation data
    tmp = list(zip(val_chunks, val_labels))
    random.shuffle(tmp)
    val_chunks, val_labels = zip(*tmp)

    # Turn into numpy arrays
    logging.info('splitting based on validation split')
    full_x_train = np.asarray(train_chunks)
    full_y_train = np.asarray(train_labels)
    x_val = np.asarray(val_chunks)
    y_val = np.asarray(val_labels)

    logging.info(f'{len(train_chunks)} total chunks to train with')
    logging.info(f'full training data: {full_x_train.shape},'
                 f'{full_y_train.shape}')
    logging.info(f'full validation data: {x_val.shape}, {y_val.shape}')

    full_arr = np.array([full_x_train,
                         full_y_train,
                         x_val,
                         y_val])

    # Save to compressed pickle to maintain ordering
    with open('chunk_data_separated_ids.pkl', 'wb') as outfile:
        pickle.dump(full_arr, outfile, pickle.HIGHEST_PROTOCOL)
