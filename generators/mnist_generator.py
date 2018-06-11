import random
import numpy as np
from scipy.ndimage.interpolation import zoom
import mnist


class MnistGenerator(object):

    def __init__(self, dims=(200, 200, 24),
                 batch_size=16, shuffle=True, validation=False,
                 extend_dims=True):
        self.dims = dims
        self.batch_size = batch_size
        self.extend_dims = extend_dims

        # Get MNIST files, Split based on validation
        if validation:
            files = mnist.test_images()
            labels = mnist.test_labels()
        else:
            files = mnist.train_images()
            labels = mnist.train_labels()

        # Take into account shuffling
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
                yield x, y

    def get_steps_per_epoch(self):
        return len(self.files) // self.batch_size

    def __data_generation(self, i):
        bsz = self.batch_size
        files = self.files[i * bsz:(i + 1) * bsz]
        labels = self.labels[i * bsz:(i + 1) * bsz]
        images = []

        for file in files:
            img = self.__transform_images(file)
            images.append(img)
        images = np.array(images)
        print("Loaded entire batch.")
        print(np.shape(images))
        return images, labels

    def __transform_images(self, image):
        # Add a third dimension
        image = np.repeat(image[:, :, np.newaxis], self.dims[2], axis=2)

        # Interpolate axis to reshape to specified dimensions
        dims = np.shape(image)
        image = zoom(image, (self.dims[0] / dims[0],
                             self.dims[1] / dims[1],
                             1))

        # Expand dims
        if self.extend_dims:
            image = np.expand_dims(image, axis=-1)
        return image
