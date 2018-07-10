import numpy as np
from keras.preprocessing import image

from generators.luke import standard_generators


def test_create_generators_same_mean_std():
    x_train = np.random.rand(500, 200, 200, 3)
    y_train = np.random.randint(0, 2, size=(500,))
    x_valid = np.random.rand(400, 200, 200, 3)
    y_valid = np.random.randint(0, 2, size=(400,))

    train_gen, valid_gen = standard_generators(x_train, y_train,
                                               x_valid, y_valid,
                                               rotation_range=20,
                                               batch_size=32)
    train_datagen: image.ImageDataGenerator = train_gen.image_data_generator
    valid_datagen: image.ImageDataGenerator = valid_gen.image_data_generator
    assert np.all(train_datagen.mean == valid_datagen.mean)
    assert np.all(train_datagen.std == valid_datagen.std)
