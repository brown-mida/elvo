from keras.layers import (
    Input,
    AveragePooling3D,
    Conv3D,
    MaxPooling3D,
    Dropout,
    Flatten
)
from keras.models import Model

CUBE_SIZE = 32


class C3DBuilder(object):

    @staticmethod
    def build(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)) -> Model:
        inputs = Input(shape=input_shape, name="input_1")
        x = inputs
        x = AveragePooling3D(pool_size=(2, 1, 1),
                             strides=(2, 1, 1), padding="same")(x)
        x = Conv3D(64, (3, 3, 3), activation='relu',
                   padding='same', name='conv1',
                   strides=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(1, 2, 2),
                         strides=(1, 2, 2), padding='valid',
                         name='pool1')(x)

        # 2nd layer group
        x = Conv3D(128, (3, 3, 3), activation='relu',
                   padding='same', name='conv2',
                   strides=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool2')(x)
        x = Dropout(rate=0.5)(x)

        # 3rd layer group
        x = Conv3D(256, (3, 3, 3), activation='relu',
                   padding='same', name='conv3a',
                   strides=(1, 1, 1))(x)
        x = Conv3D(256, (3, 3, 3), activation='relu',
                   padding='same', name='conv3b',
                   strides=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool3')(x)
        x = Dropout(rate=0.5)(x)

        # 4th layer group
        x = Conv3D(512, (3, 3, 3), activation='relu',
                   padding='same', name='conv4a',
                   strides=(1, 1, 1))(x)
        x = Conv3D(512, (3, 3, 3), activation='relu',
                   padding='same', name='conv4b',
                   strides=(1, 1, 1), )(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool4')(x)
        x = Dropout(rate=0.5)(x)

        last64 = Conv3D(64, (2, 2, 2), activation="relu",
                        name="last_64")(x)
        out_class = Conv3D(1, (1, 1, 1), activation="sigmoid",
                           name="out_class_last")(last64)
        out_class = Flatten(name="out_class")(out_class)

        model = Model(inputs=inputs, outputs=out_class)
        return model
