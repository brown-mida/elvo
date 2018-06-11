import random
import numpy as np
import mnist

from ml.generators.generator import Generator


class MnistGenerator(Generator):

    def __init__(self, dims=(120, 120, 64), batch_size=16,
                 shuffle=True,
                 validation=False,
                 split=0.2, extend_dims=True,
                 augment_data=True):
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
