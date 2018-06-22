import numpy as np

from keras import backend as K

from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from ml.generators.mip_generator import MipGenerator


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
 

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def save_features():
    model = ResNet50(weights='imagenet', include_top=False)
    gen = MipGenerator(
        dims=(200, 200, 3),
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
        dims=(200, 200, 3),
        batch_size=4,
        augment_data=False,
        extend_dims=False,
        validation=True,
        shuffle=True,
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
                  loss='binary_crossentropy', 
                  metrics=['accuracy', sensitivity, specificity])

    mc_callback = ModelCheckpoint(filepath='tmp/stage_1_resnet', 
                                  save_best_only=True,
                                  monitor='val_acc',
                                  mode='max',
                                  verbose=1)
    model.fit(train_data, train_labels,
              epochs=5000,
              batch_size=4,
              validation_data=(test_data, test_labels),
              callbacks=[mc_callback])
    model.save_weights('tmp/top_weights')


def train_top_model_2():
    train_data = np.load('tmp/features_train.npy')
    train_labels = np.load('tmp/labels_train.npy')[:1576]
    test_data = np.load('tmp/features_test.npy')
    test_labels = np.load('tmp/labels_test.npy')[:84]
    
    metrics.sensitivity = sensitivity
    metrics.specificity = specificity
    model = load_model('tmp/stage_1_resnet')
    
    model.compile(optimizer=SGD(lr=1e-6, momentum=0.9),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', sensitivity, specificity])

    mc_callback = ModelCheckpoint(filepath='tmp/stage_2_resnet', 
                                  save_best_only=True,
                                  monitor='val_acc',
                                  mode='max',
                                  verbose=1)
    model.fit(train_data, train_labels,
              epochs=500,
              batch_size=4,
              validation_data=(test_data, test_labels),
              callbacks=[mc_callback])


def fine_tune():
    model_1 = ResNet50(weights='imagenet', include_top=False)

    metrics.sensitivity = sensitivity
    metrics.specificity = specificity
    model_2 = load_model('tmp/stage_1_resnet')
    print(model_2.layers[-3:][0].get_weights())
    model = Model(input=model_1.input, output=model_2(model_1.output))

    for layer in model.layers[:141]:  # 38, 79, 141
        layer.trainable = False
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity])

    print(model.layers[-3:])
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
        epochs=20,
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


# save_features()
# train_top_model()
# train_top_model_2()
fine_tune()
# fine_tune_2()
