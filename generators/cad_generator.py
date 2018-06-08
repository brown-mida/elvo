import os
import random
import numpy as np

from preprocessing import transforms


class CadGenerator(object):

    def __init__(self, loc, batch_size=16,
                 shuffle=True, validation=False, split=0.2):
        self.loc = loc
        self.batch_size = batch_size

        filenames = []
        for filename in os.listdir(loc):
            if 'npy' in filename:
                filenames.append(filename)

        if validation:
            filenames = filenames[:int(len(filenames) * split)]
        else:
            filenames = filenames[int(len(filenames) * split):]

        if shuffle:
            random.shuffle(filenames)

        self.filenames = filenames

    def generate(self):
        steps = self.get_steps_per_epoch()
        while True:
            for i in range(steps):
                print(i)
                x = self.__data_generation(i)
                yield x, x

    def get_steps_per_epoch(self):
        return len(self.filenames) // self.batch_size

    def __data_generation(self, i):
        bsz = self.batch_size
        filenames = self.filenames[i * bsz:(i + 1) * bsz]
        images = []
        for filename in filenames:
            print("Loaded " + filename)
            images.append(np.load(self.loc + '/' + filename))
        images = self.__transform_images(images)
        return images

    def __transform_images(self, images):
        images = np.array(images)
        normalized = transforms.normalize(images)
        return np.expand_dims(normalized, axis=4)
