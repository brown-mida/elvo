import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def standard_generators(x_train: np.ndarray,
                        y_train: np.ndarray,
                        x_valid: np.ndarray,
                        y_valid: np.ndarray,
                        batch_size: int,
                        rotation_range: float,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0,
                        zoom_range=(1.0, 1.0),
                        horizontal_flip=True,
                        vertical_flip=False,
                        **kwargs):
    """
    Creates a standard training and validation generator
    from the input data.

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param rotation_range:
    :param batch_size:
    :return:
    """
    train_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       rotation_range=rotation_range,
                                       width_shift_range=width_shift_range,
                                       height_shift_range=height_shift_range,
                                       zoom_range=zoom_range,
                                       shear_range=shear_range,
                                       horizontal_flip=horizontal_flip,
                                       vertical_flip=vertical_flip)
    valid_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True)
    train_datagen.fit(x_train)
    valid_datagen.fit(x_train)

    train_gen = train_datagen.flow(x_train, y_train,
                                   batch_size=batch_size)
    valid_gen = valid_datagen.flow(x_valid, y_valid,
                                   batch_size=batch_size)
    return train_gen, valid_gen
