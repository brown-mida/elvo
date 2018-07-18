import tensorflow as tf
from models.three_d import c3d
from blueno.slack import slack_report
from blueno import utils
import os
import csv
import numpy as np
import random
from google.cloud import storage
from etl.lib import cloud_management

from keras.optimizers import SGD
from keras import backend as K

BLACKLIST = []
LEARN_RATE = 1e-5


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def false_negatives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# Delete all content in tmp/npy/
filelist = [f for f in os.listdir('tmp/npy')]
for f in filelist:
    os.remove(os.path.join('tmp/npy', f))

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
print('getting 12168 positive labels')
for id_, label in list(prelim_label_data.items()):
    if label == 1 and '_' in id_:
        positive_label_data[id_] = label

# Get 14500 random negatives from the label data to feed into our generator
negative_counter = 0
negative_label_data = {}
print("getting 14500 random negative labels")
while negative_counter < 14500:
    id_, label = random.choice(list(prelim_label_data.items()))
    if label == 0:
        negative_label_data[id_] = label
        del prelim_label_data[id_]
        negative_counter += 1

chunks = []
labels = []

i = 1
for id_, label in list(negative_label_data.items()):
    if i % 500 == 0:
        print(f'got chunk {i}')
    i += 1
    blob = bucket.get_blob('chunk_data/normal/negative/' + id_ + '.npy')
    arr = cloud_management.download_array(blob)
    if arr.shape == (32, 32, 32):
        arr = np.expand_dims(arr, axis=-1)
        chunks.append(arr)
        labels.append(label)
print(f'{i} total negative chunks')

i = 1
for id_, label in list(positive_label_data.items()):
    if i % 500 == 0:
        print(f'got chunk {i}')
    i += 1
    blob = bucket.get_blob('chunk_data/normal/positive/' + id_ + '.npy')
    arr = cloud_management.download_array(blob)
    if arr.shape == (32, 32, 32):
        arr = np.expand_dims(arr, axis=-1)
        chunks.append(arr)
        labels.append(label)
print(f'{i} total positive chunks')

tmp = list(zip(chunks, labels))
random.shuffle(tmp)
chunks, labels = zip(*tmp)

# Split based on validation
print('splitting based on validation split')
full_x_train = np.asarray(chunks[int(len(chunks) * 0.1):])
full_y_train = np.asarray(labels[int(len(labels) * 0.1):])

full_x_val = np.asarray(chunks[:int(len(chunks) * 0.1)])
full_y_val = np.asarray(labels[:int(len(labels) * 0.1)])

print(f'{len(chunks)} total chunks to train with')

print(f'full training data: {full_x_train.shape}, {full_y_train.shape}')
print(f'full validation data: {full_x_val.shape}, {full_y_val.shape}')

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

metrics = ['acc',
           utils.true_positives,
           utils.false_negatives,
           utils.sensitivity,
           utils.specificity]

for i in range(1, 11):

    model = c3d.C3DBuilder.build()
    opt = SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt,
                  loss={"out_class": "binary_crossentropy"},
                  metrics=metrics)

    frac = i / 10
    X_train = full_x_train[:int(len(full_x_train) * frac)]
    y_train = full_y_train[:int(len(full_y_train) * frac)]
    X_val = full_x_val[:int(len(full_x_val) * frac)]
    y_val = full_y_val[:int(len(full_y_val) * frac)]
    callbacks = utils.create_callbacks(x_train=X_train,
                                       y_train=y_train,
                                       x_valid=X_val,
                                       y_valid=y_val,
                                       normalize=False)

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=100,
                        batch_size=16,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        verbose=1)

    slack_report(x_train=X_train,
                 x_valid=X_val,
                 y_valid=y_val,
                 model=model,
                 history=history,
                 name=f'Basic C3D (training on {frac * 100}% of data)',
                 params=f'The most basic, non-optimized version of C3D, '
                        f'training on {frac * 100}% of data',
                 token='xoxp-314216549302'
                       '-332571517623'
                       '-402064251175'
                       '-cde67240d96f69a3534e5c919ff097e7')
