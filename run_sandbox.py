# import sandbox.generator_vis
# import sandbox.sandbox2
# import sandbox.size

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.applications.resnet50 import ResNet50
from keras.layers.convolutional import Conv2D, MaxPooling2D
from ml.generators.mip_generator import MipGenerator


def save_features():
    model = ResNet50(weights='imagenet', include_top=False)
    gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=True,
        shuffle=False
    )
    features_train = model.predict_generator(
        generator=gen.generate(),
        steps=gen.get_steps_per_epoch(),
        verbose=1
    )
    np.save('tmp/features_train.npy', features_train)
    np.save('tmp/labels_train.npy', gen.labels)

    gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=True,
        validation=True,
        shuffle=False
    )
    features_test = model.predict_generator(
        generator=gen.generate(),
        steps=gen.get_steps_per_epoch(),
        verbose=1
    )
    np.save('tmp/features_test.npy', features_test)
    np.save('tmp/labels_test.npy', gen.labels)


def train_top_model():
    train_data = np.load('tmp/features_train.npy')
    train_labels = np.load('tmp/labels_train.npy')[:700]
    test_data = np.load('tmp/features_test.npy')
    test_labels = np.load('tmp/labels_test.npy')[:172]

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels,
              epochs=10,
              batch_size=4,
              validation_data=(test_data, test_labels))
    model.save_weights('tmp/top_weights')


train_top_model()
