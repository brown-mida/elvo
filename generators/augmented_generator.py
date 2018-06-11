import os
import random

import numpy as np
import pandas as pd
import transforms
from scipy.ndimage.interpolation import zoom


class AugmentedGenerator(object):

    def __init__(self, loc, labels_loc, dims=(200, 200, 24), batch_size=16,
                 shuffle=True, validation=False, split=0.2, extend_dims=True):
        self.loc = loc
        self.dims = dims
        self.batch_size = batch_size
        self.extend_dims = extend_dims

        files = []
        blacklisted = ['patient-QPDX2K3DS7IS5QNM.npy',
                       'patient-LAUIHISOEZIM5ILF.npy',
                       'patient-NQXVKFP54XTD3GVF.npy']
        for file in os.listdir(loc):
            if ('npy' in file and
               file not in blacklisted):
                # Data augmentation
                files.append({
                    "name": file,
                    "mode": "original"
                })
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

        if validation:
            files = files[:int(len(files) * split)]
        else:
            files = files[int(len(files) * split):]

        label_data = pd.read_excel(labels_loc)
        labels = np.zeros(len(files))
        for _, row in label_data.sample(frac=1).iterrows():
            for i, file in enumerate(files):
                if row['PatientID'] == file["name"][8:-4]:
                    labels[i] = (row['ELVO status'] == 'Yes')

        if shuffle:
            tmp = list(zip(files, labels))
            random.shuffle(tmp)
            files, labels = zip(*tmp)
            labels = np.array(labels)

        self.files = files
        self.labels = labels

    def generate(self):
        steps = self.get_steps_per_epoch()
        while True:
            for i in range(steps):
                print(i)
                x, y = self.__data_generation(i)
                print(np.shape(x))
                yield x, y

    def get_steps_per_epoch(self):
        return len(self.files) // self.batch_size

    def __data_generation(self, i):
        bsz = self.batch_size
        files = self.files[i * bsz:(i + 1) * bsz]
        labels = self.labels[i * bsz:(i + 1) * bsz]
        images = []
        for file in files:
            img = np.load(self.loc + '/' + file["name"])
            img = self.__transform_images(img, file["mode"])
            images.append(img)
            print("Loaded " + file["name"])
            print(np.shape(img))
        images = np.array(images)
        return images, labels

    def __transform_images(self, image, mode):
        # Cut z axis to 200, keep x and y intact
        image = transforms.crop_z(image)
        image = np.moveaxis(image, 0, -1)

        # Data augmentation methods, cut x and y to 200x200
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
        image = transforms.crop_center(image, self.dims[0],
                                       self.dims[1])

        # Interpolate z axis to reduce to 24
        image = zoom(image, (1, 1, self.dims[2] / 200))
        image = transforms.normalize(image)
        if self.extend_dims:
            image = np.expand_dims(image, axis=-1)
        return image
