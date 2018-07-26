from keras.layers import (
    Input,
    Activation,
    Convolution3D,
    MaxPooling3D,
    Dropout,
    Flatten,
    Dense,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.regularizers import l2

CUBE_SIZE = 32
USE_DROPOUT = False
LEARN_RATE = 0.001

"""
layers:
            3D Convolution
            Leaky ReLu
            Dropout
            3d Convolution
            Leaky ReLu
            MaxPool
            Dropout
            Dense
            Dropout
            Dense
"""


class NoGenVoxNetBuilder(object):

    @staticmethod
    def build(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1),
              features=False) -> Model:
        inputs = Input(shape=input_shape, name="input_1")
        x = inputs
        # convolution 1
        x = Convolution3D(input_shape=(32, 32, 32, 1),
                          nb_filter=32,
                          kernel_dim1=5,
                          kernel_dim2=5,
                          kernel_dim3=5,
                          init='normal',
                          border_mode='valid',
                          subsample=(2, 2, 2),
                          dim_ordering='th',
                          W_regularizer=l2(0.001),
                          b_regularizer=l2(0.001),
                          )(x)

        # Activation Leaky ReLu
        x = Activation(LeakyReLU(alpha=0.1))(x)

        # dropout 1
        x = Dropout(p=0.3)(x)

        # convolution 2
        x = Convolution3D(nb_filter=32,
                          kernel_dim1=3,
                          kernel_dim2=3,
                          kernel_dim3=3,
                          init='normal',
                          border_mode='valid',
                          subsample=(1, 1, 1),
                          dim_ordering='th',
                          W_regularizer=l2(0.001),
                          b_regularizer=l2(0.001),
                          )(x)

        # Activation Leaky ReLu
        x = Activation(LeakyReLU(alpha=0.1))(x)

        # max pool 1
        x = MaxPooling3D(pool_size=(2, 2, 2),
                         strides=None,
                         border_mode='valid',
                         dim_ordering='th')(x)

        # dropout 2
        x = Dropout(p=0.4)(x)

        # dense 1 (fully connected layer)
        x = Flatten(name="out_class")(x)

        x = Dense(output_dim=128,
                  init='normal',
                  activation='linear',
                  W_regularizer=l2(0.001),
                  b_regularizer=l2(0.001),
                  )(x)

        # dropout 3
        x = Dropout(p=0.5)(x)

        # dense 2 (fully connected layer)

        nb_classes = 2  # TODO
        x = Dense(output_dim=nb_classes,
                  init='normal',
                  activation='linear',
                  W_regularizer=l2(0.001),
                  b_regularizer=l2(0.001),
                  )(x)

        # Activation Softmax
        out_class = Activation("softmax")(x)

        model = Model(input=inputs, output=out_class)
        model.summary(line_length=140)

        return model
