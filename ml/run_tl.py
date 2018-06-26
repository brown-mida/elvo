import numpy as np
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD

from ml.generators.mip_generator import MipGenerator


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def save_features():
    model = ResNet50(weights='imagenet', include_top=False)
    gen = MipGenerator(
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=True,
        extend_dims=False,
        shuffle=True,
        split=0.2
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
        test=True,
        split_test=True,
        shuffle=True,
        split=0.2
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
    train_labels = np.load('tmp/labels_train.npy')[:1404]
    test_data = np.load('tmp/features_test.npy')
    test_labels = np.load('tmp/labels_test.npy')[:172]

    inp = Input(shape=train_data.shape[1:])
    x = GlobalAveragePooling2D(name='t_pool')(inp)
    x = Dense(1024, activation='relu', name='t_dense_1')(x)
    x = Dropout(0.5, name='t_do_1')(x)
    x = Dense(1024, activation='relu', name='t_dense_2')(x)
    x = Dropout(0.5, name='t_do_2')(x)
    outp = Dense(1, activation='sigmoid', name='t_dense_3')(x)
    model = Model(input=inp, output=outp)

    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity])

    mc_callback = ModelCheckpoint(filepath='tmp/stage_1_resnet',
                                  save_best_only=True,
                                  monitor='val_acc',
                                  mode='max',
                                  verbose=1)
    mc_callback_w = ModelCheckpoint(filepath='tmp/stage_1_resnet_weights',
                                    save_best_only=True,
                                    monitor='val_acc',
                                    mode='max',
                                    save_weights_only=True,
                                    verbose=1)
    model.fit(train_data, train_labels,
              epochs=5000,
              batch_size=4,
              validation_data=(test_data, test_labels),
              callbacks=[mc_callback, mc_callback_w])
    model.save_weights('tmp/top_weights')


def fine_tune():
    model_1 = ResNet50(weights='imagenet', include_top=False)

    l1 = GlobalAveragePooling2D(name='t_pool')
    l2 = Dense(1024, activation='relu', name='t_dense_1')
    l3 = Dropout(0.5, name='t_do_1')
    l4 = Dense(1024, activation='relu', name='t_dense_2')
    l5 = Dropout(0.5, name='t_do_2')
    l6 = Dense(1, activation='sigmoid', name='t_dense_3')

    x = l1(model_1.output)
    x = l2(x)
    x = l3(x)
    x = l4(x)
    x = l5(x)
    outp = l6(x)
    model = Model(input=model_1.input, output=outp)

    print(l2.get_weights())
    model.load_weights('tmp/stage_1_resnet_weights', by_name=True)
    print(l2.get_weights())

    for layer in model.layers[:38]:  # 38, 79, 141
        layer.trainable = False
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity])

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
        shuffle=True,
        split=0.1
    )

    mc_callback = ModelCheckpoint(filepath='tmp/stage_2_resnet',
                                  save_best_only=True,
                                  monitor='val_acc',
                                  mode='max',
                                  verbose=1)

    model.fit_generator(
        generator=train_gen.generate(),
        steps_per_epoch=train_gen.get_steps_per_epoch(),
        validation_data=test_gen.generate(),
        validation_steps=test_gen.get_steps_per_epoch(),
        epochs=5000,
        verbose=1,
        callbacks=[mc_callback]
    )


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


save_features()
# train_top_model()
# train_top_model_2()
# fine_tune()
# fine_tune_2()
