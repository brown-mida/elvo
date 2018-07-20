from keras.models import Model
from keras.layers import (
    Input,
    AveragePooling3D,
    Convolution3D,
    MaxPooling3D,
    Dropout,
    Flatten
)

CUBE_SIZE = 32


class C3DBuilder(object):

    @staticmethod
    def build(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)) -> Model:
        inputs = Input(shape=input_shape, name="input_1")
        x = inputs
        x = AveragePooling3D(pool_size=(2, 1, 1),
                             strides=(2, 1, 1), border_mode="same")(x)
        x = Convolution3D(64, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv1',
                          subsample=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(1, 2, 2),
                         strides=(1, 2, 2), border_mode='valid',
                         name='pool1')(x)

        # 2nd layer group
        x = Convolution3D(128, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv2',
                          subsample=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         border_mode='valid', name='pool2')(x)
        x = Dropout(p=0.3)(x)

        # 3rd layer group
        x = Convolution3D(256, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv3a',
                          subsample=(1, 1, 1))(x)
        x = Convolution3D(256, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv3b',
                          subsample=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         border_mode='valid', name='pool3')(x)
        x = Dropout(p=0.4)(x)

        # 4th layer group
        x = Convolution3D(512, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv4a',
                          subsample=(1, 1, 1))(x)
        x = Convolution3D(512, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv4b',
                          subsample=(1, 1, 1),)(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         border_mode='valid', name='pool4')(x)
        x = Dropout(p=0.5)(x)

        last64 = Convolution3D(64, 2, 2, 2, activation="relu",
                               name="last_64")(x)
        out_class = Convolution3D(1, 1, 1, 1, activation="sigmoid",
                                  name="out_class_last")(last64)
        out_class = Flatten(name="out_class")(out_class)

        model = Model(input=inputs, output=out_class)
        model.summary(line_length=140)

        return model
