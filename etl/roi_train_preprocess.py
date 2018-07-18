import csv
import numpy as np
import random
from google.cloud import storage
from lib import cloud_management
import pickle
import logging

# Access Google Cloud Storage
gcs_client = storage.Client.from_service_account_json(
    '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'
)
bucket = gcs_client.get_bucket('elvos')

# Get label data from Google Cloud Storage
blob = storage.Blob('augmented_annotated_labels.csv', bucket)
blob.download_to_filename('tmp/augmented_annotated_labels.csv')
prelim_label_data = {}
with open('tmp/augmented_annotated_labels.csv', 'r') as pos_file:
    reader = csv.reader(pos_file, delimiter=',')
    for row in reader:
        if row[1] != 'Unnamed: 0':
            prelim_label_data[row[1]] = int(row[2])

# Get all of the positives from the label data
positive_label_data = {}
logging.info('getting 12168 positive labels')
for id_, label in list(prelim_label_data.items()):
    if label == 1 and '_' in id_:
        positive_label_data[id_] = label

# Get 14500 random negatives from the label data to feed into our generator
negative_counter = 0
negative_label_data = {}
logging.info("getting 14500 random negative labels")
while negative_counter < 14500:
    id_, label = random.choice(list(prelim_label_data.items()))
    if label == 0:
        if negative_counter % 500 == 0:
            logging.info(f'gotten {negative_counter} labels so far')
        negative_label_data[id_] = label
        del prelim_label_data[id_]
        negative_counter += 1

chunks = []
labels = []

i = 1
for id_, label in list(negative_label_data.items()):
    if i % 500 == 0:
        logging.info(f'got chunk {i}')
    i += 1
    blob = bucket.get_blob('chunk_data/normal/negative/' + id_ + '.npy')
    arr = cloud_management.download_array(blob)
    if arr.shape == (32, 32, 32):
        arr = np.expand_dims(arr, axis=-1)
        chunks.append(arr)
        labels.append(label)
logging.info(f'{i} total negative chunks')

i = 1
for id_, label in list(positive_label_data.items()):
    if i % 500 == 0:
        logging.info(f'got chunk {i}')
    i += 1
    blob = bucket.get_blob('chunk_data/normal/positive/' + id_ + '.npy')
    arr = cloud_management.download_array(blob)
    if arr.shape == (32, 32, 32):
        arr = np.expand_dims(arr, axis=-1)
        chunks.append(arr)
        labels.append(label)
logging.info(f'{i} total positive chunks')

tmp = list(zip(chunks, labels))
random.shuffle(tmp)
chunks, labels = zip(*tmp)

# Split based on validation
logging.info('splitting based on validation split')
full_x_train = np.asarray(chunks[int(len(chunks) * 0.1):])
full_y_train = np.asarray(labels[int(len(labels) * 0.1):])

full_x_val = np.asarray(chunks[:int(len(chunks) * 0.1)])
full_y_val = np.asarray(labels[:int(len(labels) * 0.1)])

logging.info(f'{len(chunks)} total chunks to train with')

logging.info(f'full training data: {full_x_train.shape}, {full_y_train.shape}')
logging.info(f'full validation data: {full_x_val.shape}, {full_y_val.shape}')

full_arr = np.array([full_x_train,
                     full_y_train,
                     full_x_val,
                     full_y_val])

with open('chunk_data.pkl', 'wb') as outfile:
    pickle.dump(full_arr, outfile, pickle.HIGHEST_PROTOCOL)
