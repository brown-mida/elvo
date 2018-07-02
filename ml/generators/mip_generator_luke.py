import os
import csv
import random
import numpy as np
from scipy.ndimage.interpolation import zoom
from keras.preprocessing.image import ImageDataGenerator

from google.cloud import storage
from lib import transforms

BLACKLIST = ['LAUIHISOEZIM5ILF',
             '2018050121043822',
             '2018050120260258']


class MipGenerator(object):

    def __init__(self, dims=(120, 120, 1), batch_size=16,
                 shuffle=True,
                 validation=False,
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
            zoom_range=0.1,
            horizontal_flip=True
        )

        # Delete all content in tmp/npy/
        filelist = [f for f in os.listdir('tmp/npy')]
        for f in filelist:
            os.remove(os.path.join('tmp/npy', f))

        # Get npy files from Google Cloud Storage
        gcs_client = storage.Client.from_service_account_json(
            'credentials/client_secret.json'
        )
        bucket = gcs_client.get_bucket('elvos')

        files = []
        blobs = bucket.list_blobs(
            prefix='multichannel_mip_data/from_luke_training/'
        )
        for blob in blobs:
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

                if self.augment_data and not self.validation:
                    self.__add_augmented(files, file)

        blobs = bucket.list_blobs(
            prefix='multichannel_mip_data/from_luke_validation/'
        )
        for blob in blobs:
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

                if self.augment_data and not self.validation:
                    self.__add_augmented(files, file)

        # Split based on validation
        if validation:
            files = files[:int(len(files) * split)]
        else:
            files = files[int(len(files) * split):]

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
            filename = filename.split('/')[-1]
            filename = filename.split('.')[0]
            filename = filename.split('_')[0]
            labels[i] = label_data[filename]

        # Take into account shuffling
        if shuffle:
            tmp = list(zip(files, labels))
            random.shuffle(tmp)
            files, labels = zip(*tmp)
            labels = np.array(labels)

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
        while True:
            for i in range(steps):
                print(i)
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
            blob = self.bucket.get_blob(file['name'])
            file_id = file['name'].split('/')[-1]
            file_id = file_id.split('.')[0]
            blob.download_to_filename(
                'tmp/npy/{}.npy'.format(file_id)
            )
            img = np.load('tmp/npy/{}.npy'.format(file_id))
            os.remove('tmp/npy/{}.npy'.format(file_id))
            img = self.__transform_images(img)
            # print(np.shape(img))
            images.append(img)
        images = np.array(images)
        print("Loaded entire batch.")
        print(np.shape(images))
        return images, labels

    def __transform_images(self, image):
        image = np.moveaxis(image, 0, -1)

        # Set bounds
        image[image < -40] = -40
        image[image > 400] = 400

        # Normalize image and expand dims
        image = transforms.normalize(image)
        if self.extend_dims:
            if len(self.dims) == 2:
                image = np.expand_dims(image, axis=-1)
            else:
                image = np.repeat(image[:, :, np.newaxis],
                                  self.dims[2], axis=2)

        # Data augmentation methods
        if self.augment_data and not self.validation:
            image = self.datagen.random_transform(image)

        # Interpolate axis to reduce to specified dimensions
        dims = np.shape(image)
        image = zoom(image, (self.dims[0] / dims[0],
                             self.dims[1] / dims[1],
                             1))
        return image
