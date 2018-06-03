import os
import random
import numpy as np
import pandas as pd
import scipy.ndimage

from preprocessing import transforms


class Generator(object):

    def __init__(self, loc, labels_loc, dim_length=64, batch_size=16,
                 shuffle=True, validation=False, split=0.2):
        self.loc = loc
        self.dim_length = dim_length
        self.batch_size = batch_size
        
        filenames = []
        for filename in os.listdir(loc):
            if 'npy' in filename:
                filenames.append(filename)

        if validation:
            filenames = filenames[:int(len(filenames) * split)]
        else:
            filenames = filenames[int(len(filenames) * split):]

        label_data = pd.read_excel(labels_loc)
        labels = np.zeros(len(filenames))
        for _, row in label_data.sample(frac=1).iterrows():
            for i, id_ in enumerate(filenames):
                if row['PatientID'] == id_[8:-4]:
                    labels[i] = (row['ELVO status'] == 'Yes')

        if shuffle:
            tmp = list(zip(filenames, labels))
            random.shuffle(tmp)
            filenames, labels = zip(*tmp)
            labels = np.array(labels)

        self.filenames = filenames
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
        return len(self.filenames) // self.batch_size


    def __data_generation(self, i):
        bsz = self.batch_size
        filenames = self.filenames[i * bsz:(i + 1) * bsz]
        labels = self.labels[i * bsz:(i + 1) * bsz]
        images = []
        for filename in filenames:
            images.append(np.load(self.loc + '/' + filename))
            print("Loaded " + filename)
        images = self.__transform_images(images, self.dim_length)
        return images, labels

    def __transform_images(self,images, dim_length):
        resized = np.stack([scipy.ndimage.interpolation.zoom(arr, (24 / 200, 1, 1))
                    for arr in images])
        resized = np.moveaxis(resized, 1, -1)
        normalized = transforms.normalize(resized)
        return np.expand_dims(normalized, axis=4)




