"""
A simple 3D Convolutional network. It has 5 convolutional
layers and 3 FC layers, much like Alexnet.

Unfortunately, the architecture described in the paper is
severely lacking in details. Here I use 2D convolution
(with depth as channels) instead of 3D. In addition, I assume:
- The filter sizes
- The strides between convolutional layers
- The pooling size

Based on:
- https://github.com/ehosseiniasl/3d-convolutional-network
- https://www.nature.com/articles/s41746-017-0015-z.pdf
"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


class SimpleNetBuilder(object):

    @staticmethod
    def build(input_shape):
        """Create a Simple CNN  model.
        """

        if len(input_shape) != 3:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3)")

        input_img = Input(shape=input_shape, name="cad_input")

        # Conv1 (Output 200 x 200 x 48)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same')(input_img)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same')(x)
        x = MaxPooling2D()(x)

        # Conv2 (Output 50 x 50 x 64)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        # Flatten
        x = Flatten()(x)

        # Fully connected layers
        output_img = Dense(1, activation='softmax', use_bias=True)(x)

        model = Model(inputs=input_img, outputs=output_img)
        return model


# m = SimpleNetBuilder.build((120, 120, 64))
# m.summary()
