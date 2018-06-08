import os
import csv
import random
import numpy as np
from scipy.ndimage.interpolation import zoom

from google.cloud import storage
from preprocessing import transforms

BLACKLIST = ['LAUIHISOEZIM5ILF']


class AlexNetGenerator(object):

    def __init__(self, dims=(200, 200, 24),
                 batch_size=16, shuffle=True, validation=False,
                 split=0.2, extend_dims=True,
                 augment_data=True):
        self.dims = dims
        self.batch_size = batch_size
        self.extend_dims = extend_dims
        self.augment_data = augment_data

        # Get npy files from Google Cloud Storage
        gcs_client = storage.Client.from_service_account_json(
            'credentials/client_secret.json'
        )
        bucket = gcs_client.get_bucket('elvos')
        blobs = bucket.list_blobs(prefix='numpy/')

        files = []
        for blob in blobs:
            file = blob.name

            # Check blacklist
            if file in BLACKLIST:
                continue

            # Add all data augmentation methods
            files.append({
                "name": file,
                "mode": "original"
            })

            if self.augment_data:
                for i in range(5):
                    files.append({
                        "name": file,
                        "mode": "translate"
                    })
                for i in range(5):
                    files.append({
                        "name": file,
                        "mode": "rotate"
                    })
                for i in range(3):
                    files.append({
                        "name": file,
                        "mode": "zoom"
                    })
                for i in range(3):
                    files.append({
                        "name": file,
                        "mode": "gaussian"
                    })
                files.append({
                    "name": file,
                    "mode": "flip"
                })

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

        # Delete all content in tmp/npy/
        filelist = [f for f in os.listdir('tmp/npy')]
        for f in filelist:
            os.remove(os.path.join('tmp/npy', f))

        # Download files to tmp/npy/
        for i, file in enumerate(files):
            print("Loading " + file['name'])
            blob = self.bucket.get_blob(file['name'])
            file_id = file['name'].split('/')[-1]
            file_id = file_id.split('.')[0]
            blob.download_to_filename('tmp/npy/{}.npy'.format(file_id))
            img = np.load('tmp/npy/{}.npy'.format(file_id))
            img = self.__transform_images(img, file['mode'])
            images.append(img)
            print("Loaded " + file['name'])
            print(np.shape(img))
        images = np.array(images)
        print("Loaded entire batch.")
        print(np.shape(images))
        return images, labels

    def __transform_images(self, image, mode):
        # Cut z axis to 200, keep x and y intact
        image[image < -40] = 0
        image[image > 400] = 0
        image = np.moveaxis(image, 0, -1)
        image = np.flip(image, 2)
        image = transforms.crop_z(image, 150)

        # # Data augmentation methods, cut x and y to 80%
        if mode == "translate":
            image = transforms.translated_img(image)
        elif mode == "rotate":
            image = transforms.rotate_img(image)
        elif mode == "zoom":
            image = transforms.zoom_img(image)
        elif mode == "gaussian":
            image = transforms.gaussian_img(image)
        elif mode == "flip":
            image = transforms.flip_img(image)
        dims = np.shape(image)
        image = transforms.crop_center(image, int(dims[0] * 0.8),
                                       int(dims[1] * 0.8))

        # # Interpolate axis to reduce to specified dimensions
        dims = np.shape(image)
        image = zoom(image, (self.dims[0] / dims[0],
                             self.dims[1] / dims[1],
                             self.dims[2] / dims[2]))

        # # Normalize image and expand dims
        image = transforms.normalize(image)
        if self.extend_dims:
            image = np.expand_dims(image, axis=-1)
        return image
