import os
import csv
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from google.cloud import storage

BLACKLIST = ['LAUIHISOEZIM5ILF',
             '2018050121043822',
             '2018050120260258']


class MipGenerator(object):

    def __init__(self, data_loc='',
                 dims=(120, 120, 1), batch_size=16,
                 shuffle=True,
                 validation=False,
                 test=False, split_test=False,
                 split=0.2, extend_dims=True,
                 augment_data=True):
        self.data_loc = data_loc
        self.dims = dims
        self.batch_size = batch_size
        self.extend_dims = extend_dims
        self.augment_data = augment_data
        self.validation = validation

        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=[1.0, 1.1],
            horizontal_flip=True
        )

        # Access Google Cloud Storage
        gcs_client = storage.Client.from_service_account_json(
            'credentials/client_secret.json'
        )
        bucket = gcs_client.get_bucket('elvos')

        # Get file list
        filelist = sorted([f for f in os.listdir(data_loc)])
        files = []
        for file in filelist:
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

                if self.augment_data and not self.validation:
                    self.__add_augmented(files, file)

        # Get label data from Google Cloud Storage
        blob = storage.Blob('labels.csv', bucket)
        blob.download_to_filename('tmp/labels.csv')
        label_data = {}
        with open('tmp/labels.csv', 'r') as pos_file:
            reader = csv.reader(pos_file, delimiter=',')
            for row in reader:
                if row[0] != 'patient_id':
                    label_data[row[0]] = int(row[1])

        labels = np.zeros(len(files))
        for i, file in enumerate(files):
            filename = file['name']
            filename = filename.split('_')[0]
            filename = filename.split('.')[0]
            labels[i] = label_data[filename]

        # Take into account shuffling
        if shuffle:
            tmp = list(zip(files, labels))
            random.Random(192382491).shuffle(tmp)
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

    def __add_augmented(self, files, file):
        for i in range(1):
            files.append({
                "name": file,
            })

    def generate(self):
        steps = self.get_steps_per_epoch()
        print(steps)
        while True:
            for i in range(steps):
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
            img = np.load('{}/{}.npy'.format(self.data_loc, file_id))
            img = self.__transform_images(img)
            images.append(img)
        images = np.array(images)
        return images, labels

    def __transform_images(self, image):
        # Data augmentation methods
        if self.augment_data and not self.validation:
            image = self.datagen.random_transform(image)
        return image
