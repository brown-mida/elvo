import numpy as np
import pytest
from keras.preprocessing import image

import generators.luke
import utils


def test_upload_to_slack():
    with open('test_upload_to_slack.png', 'w') as f:
        f.write('hello!')
    r = utils.upload_to_slack('test_upload_to_slack.png', 'just testing you')
    assert r.status_code == 200


# TODO
@pytest.mark.skip
def test_sensitivity():
    raise NotImplementedError()


@pytest.mark.skip
def test_specificity():
    raise NotImplementedError()


def test_create_generators_same_mean_std():
    x_train = np.random.rand(500, 200, 200, 3)
    y_train = np.random.randint(0, 2, size=(500,))
    x_valid = np.random.rand(400, 200, 200, 3)
    y_valid = np.random.randint(0, 2, size=(400,))

    train_gen, valid_gen = generators.luke.standard_generators(x_train, y_train,
                                                               x_valid, y_valid,
                                                               rotation_range=20,
                                                               batch_size=32)
    train_datagen: image.ImageDataGenerator = train_gen.image_data_generator
    valid_datagen: image.ImageDataGenerator = valid_gen.image_data_generator
    assert np.all(train_datagen.mean == valid_datagen.mean)
    assert np.all(train_datagen.std == valid_datagen.std)
