import csv
import numpy as np
import random
from google.cloud import storage
from lib import cloud_management
import pickle
import logging


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


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
with open('tmp/augmented_annotated_labels.csv', 'r') as pos_file:
    reader = csv.reader(pos_file, delimiter=',')
    for row in reader:
        if row[1] != 'Unnamed: 0':
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

for i, id_ in enumerate(list(positive_label_data.keys())):
    if i % 24 == 0:
        seed = random.randint(1, 100)
        stripped_id = id_[:-1]
        if seed > 10:
            positive_train_label_data[id_] = 1
            for j in range(2, 25):
                positive_train_label_data[stripped_id + str(j)] = 1
        else:
            positive_val_label_data[id_] = 1
            for j in range(2, 25):
                positive_val_label_data[stripped_id + str(j)] = 1

# Get 14500 random negatives from the label data to feed into our generator
negative_counter = 0
negative_train_label_data = {}
negative_val_label_data = {}
logging.info("getting 14500 random negative labels")
while negative_counter < 14500:
    id_, label = random.choice(list(prelim_label_data.items()))
    if label == 0:
        if negative_counter % 500 == 0:
            logging.info(f'gotten {negative_counter} labels so far')

        seed = random.randint(1, 100)
        if seed > 10:
            negative_train_label_data[id_] = label
        else:
            negative_val_label_data[id_] = label
        del prelim_label_data[id_]
        negative_counter += 1

train_chunks = []
train_labels = []
val_chunks = []
val_labels = []

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

tmp = list(zip(train_chunks, train_labels))
random.shuffle(tmp)
train_chunks, train_labels = zip(*tmp)

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
logging.info(f'full training data: {full_x_train.shape}, {full_y_train.shape}')
logging.info(f'full validation data: {x_val.shape}, {y_val.shape}')

full_arr = np.array([full_x_train,
                     full_y_train,
                     x_val,
                     y_val])

with open('chunk_data_separated.pkl', 'wb') as outfile:
    pickle.dump(full_arr, outfile, pickle.HIGHEST_PROTOCOL)
