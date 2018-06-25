import numpy as np

from keras import backend as K

from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from ml.generators.mip_generator import MipGenerator


def make_top_model():

    inp = Input(shape=(7, 7, 2048))
    x = Dense(1024, activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outp = Dense(1, activation='sigmoid')(x)

    model = Model(input=inp, output=outp)
    model.save('sandbox/test_model')


def fine_tune():
    model_1 = ResNet50(weights='imagenet', include_top=False)
    model_2 = load_model('sandbox/test_model')
    model = Model(input=model_1.input, output=model_2(model_1.output))
    model.summary()


# make_top_model()
fine_tune()
