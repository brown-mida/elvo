import os
import csv
import random
import numpy as np
from scipy.ndimage.interpolation import zoom

from google.cloud import storage
from preprocessing import transforms


class Generator(object):

    def __init__(self, dims=(200, 200, 24),
                 batch_size=16, shuffle=True, validation=False,
                 split=0.2, extend_dims=True):
        self.dims = dims
        self.batch_size = batch_size
        self.extend_dims = extend_dims

        # Get npy filenames from Google Cloud Storage
        gcs_client = storage.Client.from_service_account_json(
            'credentials/client_secret.json'
        )
        bucket = gcs_client.get_bucket('elvos')
        blobs = bucket.list_blobs(prefix='numpy/')
        filenames = []
        for blob in blobs:
            filenames.append(blob.name)

        # Split based on validation
        if validation:
            filenames = filenames[:int(len(filenames) * split)]
        else:
            filenames = filenames[int(len(filenames) * split):]

        # Get label data from Google Cloud Storage
        blob = storage.Blob('labels.csv', bucket)
        blob.download_to_filename('tmp/labels.csv')
        label_data = {}
        with open('tmp/labels.csv', 'r') as pos_file:
            reader = csv.reader(pos_file, delimiter=',')
            for row in reader:
                if row[0] != 'patient_id':
                    label_data[row[0]] = int(row[1])

        labels = np.zeros(len(filenames))
        for i, filename in enumerate(filenames):
            filename = filename.split('/')[-1]
            filename = filename.split('.')[0]
            labels[i] = label_data[filename]

        # Take into account shuffling
        if shuffle:
            tmp = list(zip(filenames, labels))
            random.shuffle(tmp)
            filenames, labels = zip(*tmp)
            labels = np.array(labels)

        self.filenames = filenames
        self.labels = labels
        self.bucket = bucket

    def generate(self):
        steps = self.get_steps_per_epoch()
        while True:
            for i in range(steps):
                print(i)
                x, y = self.__data_generation(i)
                yield x, y

    def get_steps_per_epoch(self):
        return len(self.filenames) // self.batch_size

    def __data_generation(self, i):
        bsz = self.batch_size
        filenames = self.filenames[i * bsz:(i + 1) * bsz]
        labels = self.labels[i * bsz:(i + 1) * bsz]
        images = []

        # Delete all content in tmp/npy/
        filelist = [f for f in os.listdir('tmp/npy')]
        for f in filelist:
            os.remove(os.path.join('tmp/npy', f))

        # Download files to tmp/npy/
        for i, filename in enumerate(filenames):
            blob = self.bucket.get_blob(filename)
            blob.download_to_filename('tmp/npy/{}.npy'.format(i))
            img = np.load('tmp/npy/{}.npy'.format(i))
            img = self.__transform_images(img)
            images.append(img)
            print("Loaded " + filename)
        print("Loaded entire batch.")
        print(np.shape(images))
        return images, labels

    def __transform_images(self, image):
        # TODO: Ideally we want all data downloaded to be the same size.
        # Since they are not all at the same size, we have to make some
        # concessions.

        # Cut z axis to 200, keep x and y intact
        # image = transforms.crop_z(image)
        image = np.moveaxis(image, 0, -1)
        image = transforms.crop_center(image, self.dims[0],
                                       self.dims[1])

        # Interpolate z axis to reduce to 24
        image = zoom(image, (1, 1, self.dims[2] / np.shape(image)[2]))
        image = transforms.normalize(image)
        if self.extend_dims:
            image = np.expand_dims(image, axis=-1)
        return image
