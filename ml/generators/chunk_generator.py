import csv
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from google.cloud import storage

BLACKLIST = []


class ChunkGenerator(object):
    def __init__(self, dims=(32, 32, 32), batch_size=16,
                 shuffle=True,
                 validation=False,
                 test=False, split_test=False,
                 split=0.2, extend_dims=True,
                 augment_data=True):
        self.dims = dims
        self.batch_size = batch_size
        self.extend_dims = extend_dims
        self.augment_data = augment_data
        self.validation = validation

        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=[1.0, 1.1],
        )

        # Access Google Cloud Storage
        gcs_client = storage.Client.from_service_account_json(
            'credentials/client_secret.json'
        )
        bucket = gcs_client.get_bucket('elvos')

        # Get label data from Google Cloud Storage
        blob = storage.Blob('annotated_labels.csv', bucket)
        blob.download_to_filename('tmp/annotated_labels.csv')
        prelim_label_data = {}
        with open('tmp/annotated_labels.csv', 'r') as pos_file:
            reader = csv.reader(pos_file, delimiter=',')
            for row in reader:
                if row[1] != 'labels':
                    prelim_label_data[row[0]] = int(row[1])

        # Get 8000 random negatives from the label data to feed
        # into our generator
        negative_counter = 0
        negative_label_data = {}
        while negative_counter < 12096:
            id_, label = random.choice(list(prelim_label_data.items()))
            if label == 0:
                negative_label_data[id_] = label
                del label[id_]
                negative_counter += 1

        # Get all of the positives from the label data
        label_data = {}
        for id_, label in list(prelim_label_data.items()):
            if label == 1:
                label_data[id_] = label

        # Put pos and neg together into one dictionary
        label_data.update(negative_label_data)

        # Get positives chunks
        pos_blobs = bucket.list_blobs(prefix='chunk_data/normal/positive')
        files = []
        for blob in pos_blobs:
            file = blob.name

            # Check blacklist
            blacklisted = False
            for each in BLACKLIST:
                if each in file:
                    blacklisted = True

            if not blacklisted:
                # Add all data augmentation methods
                files.append({
                    "name": file,
                })

        # Get negative chunks that were chosen to be in the labels
        neg_blobs = bucket.list_blobs(prefix='chunk_data/normal/negative')
        for blob in neg_blobs:
            file = blob.name

            # Check blacklist
            blacklisted = False
            for each in BLACKLIST:
                if each in file:
                    blacklisted = True

            file_id = blob.name.split('/')[-1]
            file_id = file_id.split('.')[0]

            if file_id in negative_label_data and not blacklisted:
                files.append({
                    "name": file,
                })

        # convert labels from dict to np array
        labels = np.zeros(len(files))
        for i, file in enumerate(files):
            filename = file['name']
            filename = filename.split('_')[0]
            labels[i] = label_data[filename]

        # Take into account shuffling
        if shuffle:
            tmp = list(zip(files, labels))
            random.shuffle(tmp)
            files, labels = zip(*tmp)
            labels = np.array(labels)

        # Split based on validation
        if validation:
            if split_test:
                files = files[:int(len(files) * split / 2)]
                labels = labels[:int(len(labels) * split / 2)]
            else:
                files = files[:int(len(files) * split)]
                labels = labels[:int(len(labels) * split)]
        elif test:
            if split_test:
                files = files[int(len(files) * split / 2):
                              int(len(files) * split)]
                labels = labels[int(len(labels) * split / 2):
                                int(len(labels) * split)]
            else:
                raise ValueError('must set split_test to True if test')
        else:
            files = files[int(len(files) * split):]
            labels = labels[int(len(labels) * split):]
        print(np.shape(files))
        print(np.shape(labels))
        print("Negatives: {}".format(np.count_nonzero(labels == 0)))
        print("Positives: {}".format(np.count_nonzero(labels)))
        self.files = files
        self.labels = labels
        self.bucket = bucket

    def generate(self):
        steps = self.get_steps_per_epoch()
        # print(steps)
        while True:
            for i in range(steps):
                # print(i)
                # print("D")
                x, y = self.__data_generation(i)
                yield x, y

    def get_steps_per_epoch(self):
        return len(self.files) // self.batch_size

    def __data_generation(self, i):
        bsz = self.batch_size
        files = self.files[i * bsz:(i + 1) * bsz]
        labels = self.labels[i * bsz:(i + 1) * bsz]
        images = []

        # Download files to tmp/npy/
        for i, file in enumerate(files):
            file_id = file['name'].split('/')[-1]
            file_id = file_id.split('.')[0]
            print(file_id)
            img = np.load('tmp/npy/{}.npy'.format(file_id))
            # print(np.shape(img))
            images.append(img)
        images = np.array(images)
        # print("Loaded entire batch.")
        # print(np.shape(images))
        return images, labels
