# import sandbox.generator_vis
# import sandbox.sandbox2
# import sandbox.size

import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from ml.generators.multichannel_mip_generator import MipGenerator


def save_features():
    model = ResNet50(weights='imagenet', include_top=False)
    gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=True,
        extend_dims=False,
        shuffle=True,
        split=0.1
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
        extend_dims=False,
        validation=True,
        shuffle=False,
        split=0.1
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
    print(np.shape(train_data))
    train_labels = np.load('tmp/labels_train.npy')[:1576]
    test_data = np.load('tmp/features_test.npy')
    test_labels = np.load('tmp/labels_test.npy')[:84]

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels,
              epochs=3,
              batch_size=4,
              validation_data=(test_data, test_labels))
    model.save_weights('tmp/top_weights')


def fine_tune():
    model_1 = ResNet50(weights='imagenet', include_top=False)

    model_2 = Sequential()
    model_2.add(
        GlobalAveragePooling2D(input_shape=model_1.output_shape[1:], name='a')
    )
    model_2.add(Dense(1024, activation='relu', name='b'))
    model_2.add(Dropout(0.5, name='c'))
    model_2.add(Dense(1024, activation='relu'))
    model_2.add(Dropout(0.5))
    model_2.add(Dense(1, activation='sigmoid', name='d'))
    model_2.load_weights('tmp/top_weights')

    model = Model(input=model_1.input, output=model_2(model_1.output))

    for layer in model.layers[:39]:  # 38, 79, 141
        layer.trainable = False
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=16,
        augment_data=True,
        extend_dims=False,
        shuffle=True,
        split=0.1
    )

    test_gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=False,
        validation=True,
        shuffle=False,
        split=0.1
    )

    model.fit_generator(
        generator=train_gen.generate(),
        steps_per_epoch=train_gen.get_steps_per_epoch(),
        validation_data=test_gen.generate(),
        validation_steps=test_gen.get_steps_per_epoch(),
        epochs=20,
        verbose=1
    )

    model.save('tmp/trained_resnet_a')


def fine_tune_2():
    model = load_model('tmp/trained_resnet')

    for layer in model.layers[:39]:
        layer.trainable = False
    model.compile(optimizer=SGD(lr=1e-5, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=16,
        augment_data=True,
        extend_dims=False,
        shuffle=False
    )

    test_gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=False,
        validation=True,
        shuffle=False
    )

    model.fit_generator(
        generator=train_gen.generate(),
        steps_per_epoch=train_gen.get_steps_per_epoch(),
        validation_data=test_gen.generate(),
        validation_steps=test_gen.get_steps_per_epoch(),
        epochs=500,
        verbose=1
    )

    model.save('tmp/trained_resnet_2')


# save_features()
# train_top_model()
# fine_tune()
fine_tune_2()
