import numpy as np

from keras import backend as K

import keras.metrics as metrics
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
# from ml.generators.mip_generator_local import MipGenerator
from ml.generators.mip_generator_memory import MipGenerator

import ml.utils as utils


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model
    
    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
                                                                    '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def save_features(data_loc):
    model = ResNet50(weights='imagenet', include_top=False)
    gen = MipGenerator(
        data_loc=data_loc,
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
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
        data_loc=data_loc,
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=False,
        validation=True,
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
    train_labels = np.load('tmp/labels_train.npy')[:len(train_data)]
    test_data = np.load('tmp/features_test.npy')
    test_labels = np.load('tmp/labels_test.npy')[:len(test_data)]

    inp = Input(shape=train_data.shape[1:])
    x = GlobalAveragePooling2D(name='t_pool')(inp)
    x = Dense(1024, activation='relu', 
              name='t_dense_1')(x)
    x = Dropout(0.5, name='t_do_1')(x)
    x = Dense(1024, activation='relu',
              name='t_dense_2')(x)
    x = Dropout(0.5, name='t_do_2')(x)
    outp = Dense(1, activation='sigmoid', name='t_dense_3')(x)
    model = Model(input=inp, output=outp)

    model.compile(optimizer=Adam(lr=1e-4),
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


def fine_tune(data_loc, comment, img_gen=None):
    model_1 = ResNet50(weights='imagenet', include_top=False)
    # for layer in model_1.layers[:38]:
    #     layer.trainable = False

    l1 = GlobalAveragePooling2D(name='t_pool')
    l2 = Dense(1024, activation='relu', name='t_dense_1')
    l3 = Dropout(0.8, name='t_do_1')
    l4 = Dense(1024, activation='relu', name='t_dense_2')
    l5 = Dropout(0.8, name='t_do_2')
    l6 = Dense(1, activation='sigmoid', name='t_dense_3')

    x = l1(model_1.layers[141].output)
    x = l2(x)
    x = l3(x)
    x = l4(x)
    x = l5(x)
    outp = l6(x)
    model = Model(inputs=model_1.input, outputs=outp)

    # model.load_weights('tmp/stage_1_resnet_weights', by_name=True)
    # for layer in model.layers[:141]:  # 38, 79, 141
    #     layer.trainable = False
    gpu_model = ModelMGPU(model, 2)
    gpu_model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity])

    train_gen = MipGenerator(
        data_loc=data_loc,
        dims=(220, 220, 3),
        batch_size=48,
        augment_data=True,
        extend_dims=False,
        shuffle=True,
        split=0.2,
        img_gen=img_gen
    )

    test_gen = MipGenerator(
        data_loc=data_loc,
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=False,
        validation=True,
        split_test=True,
        shuffle=True,
        split=0.2
    )

    mc_callback = ModelCheckpoint(filepath='tmp/stage_2_resnet',
                                  save_best_only=True,
                                  monitor='val_acc',
                                  mode='max',
                                  verbose=1)

    history = gpu_model.fit_generator(
        generator=train_gen.generate(),
        steps_per_epoch=train_gen.get_steps_per_epoch(),
        validation_data=test_gen.generate(),
        validation_steps=test_gen.get_steps_per_epoch(),
        epochs=600,
        verbose=1,
        callbacks=[mc_callback]
    )

    train_files = np.array([x['img'] for x in train_gen.files])
    test_files = np.array([x['img'] for x in test_gen.files])
    train_labels = np.array(train_gen.labels)
    test_labels = np.array(test_gen.labels)

    utils.slack_report(train_files, train_labels,
        test_files, test_labels,
        model, history, 'Report', comment,
        'xoxp-314216549302-332571253111-395866627814-3c4d6241fdb752f29fdf328410b8384a'
    )


def fine_tune_2():
    metrics.sensitivity = sensitivity
    metrics.specificity = specificity

    model = load_model('tmp/stage_2_resnet')
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers[:141]:
        layer.trainable = False
    gpu_model = ModelMGPU(model, 4)
    gpu_model.compile(optimizer=SGD(lr=1e-5, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity])

    train_gen = MipGenerator(
        data_loc='data/mip',
        dims=(220, 220, 3),
        batch_size=48,
        augment_data=True,
        extend_dims=False,
        shuffle=True,
        split=0.2
    )

    test_gen = MipGenerator(
        data_loc='data/mip',
        dims=(220, 220, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=False,
        validation=True,
        split_test=True,
        shuffle=True,
        split=0.2
    )

    mc_callback = ModelCheckpoint(filepath='tmp/stage_3_resnet',
                                  save_best_only=True,
                                  monitor='val_acc',
                                  mode='max',
                                  verbose=1)

    gpu_model.fit_generator(
        generator=train_gen.generate(),
        steps_per_epoch=train_gen.get_steps_per_epoch(),
        validation_data=test_gen.generate(),
        validation_steps=test_gen.get_steps_per_epoch(),
        epochs=5000,
        verbose=1,
        callbacks=[mc_callback]
    )


# save_features('data/vessel_0_transform')
# train_top_model()
# train_top_model_2()
for rotation in [15, 20, 25, 30, 35, 40, 45]:
    for zoom in [[1.0, 1.1], [0.9, 1.1]]:
        for shear in [0, 15, 20, 25]:
            datagen = ImageDataGenerator(
                rotation_range=rotation,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=zoom,
                shear_range=shear,
                horizontal_flip=True
            )
            params = {
                "rotation": rotation,
                "zoom": zoom,
                "shear": shear
            }
            fine_tune('data/vessel_0', params, img_gen=datagen)
# fine_tune('data/vessel_0')
# fine_tune_2()
