import numpy as np
import pandas as pd
from keras.preprocessing import image

import resnet_train


def test_create_generators_same_mean_std():
    x_train = np.random.rand(500, 200, 200, 3)
    y_train = np.random.randint(0, 2, size=(500,))
    x_valid = np.random.rand(400, 200, 200, 3)
    y_valid = np.random.randint(0, 2, size=(400,))

    params = {
        'rotation_range': 20,
        'batch_size': 32,
    }
    train_gen, valid_gen = resnet_train.create_generators(x_train, y_train,
                                                          x_valid, y_valid,
                                                          params=params)
    train_datagen: image.ImageDataGenerator = train_gen.image_data_generator
    valid_datagen: image.ImageDataGenerator = valid_gen.image_data_generator
    assert np.all(train_datagen.mean == valid_datagen.mean)
    assert np.all(train_datagen.std == valid_datagen.std)


def test_to_shuffled_arrays():
    x_dict = {
        'a': np.array([1]),
        'b': np.array([2]),
        'c': np.array([3]),
    }
    y_df = pd.DataFrame(
        data=np.array([1, 2, 3]),
        index=['a', 'b', 'c'],
    )

    x_arr, y_arr = resnet_train.to_arrays(x_dict, y_df)
    assert np.all(x_arr == y_arr)


def test_sensitivity():
    raise NotImplementedError()


def test_specificity():
    raise NotImplementedError()
