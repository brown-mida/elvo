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
from keras.layers import (
    Input, Dense, Conv2DTranspose,
    Concatenate, Cropping2D
)
from keras.layers.convolutional import Conv2D, MaxPooling2D


class SimpleUNetBuilder(object):

    @staticmethod
    def build(input_shape, num_classes=2):
        """Create a 3D Convolutional Autoencoder model.

        Parameters:
        - input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels)
        - initial_filter: Initial filter size. This will be doubled
            for each hidden layer as it goes deeper.
        - num_encoding_layers: Number of encoding convolutional +
            pooling layers. The number of decoding
            layers will be the same.

        Returns:
        - A 3D CAD model that takes a 5D tensor (volumetric images
        in batch) as input and returns a 5D vector (prediction) as output.
        """

        if len(input_shape) != 3:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3)")

        input_img = Input(shape=input_shape, name="cad_input")

        # Conv1 (Output n, n, 96)
        conv1 = Conv2D(96, (5, 5), activation='relu',
                       padding='same')(input_img)
        conv1 = Conv2D(96, (5, 5), activation='relu',
                       padding='same')(conv1)

        # Conv2 (Output n/2, n/2, 128)
        conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = Conv2D(128, (5, 5), activation='relu',
                       padding='same')(conv2)
        conv2 = Conv2D(128, (5, 5), activation='relu',
                       padding='same')(conv2)

        # Conv3 (Output n/4, n/4, 256)
        conv3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv3 = Conv2D(256, (3, 3), activation='relu',
                       padding='same')(conv3)
        conv3 = Conv2D(256, (3, 3), activation='relu',
                       padding='same')(conv3)

        # Conv4 (Output n/8, n/8, 512)
        conv4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation='relu',
                       padding='same')(conv4)
        conv4 = Conv2D(512, (3, 3), activation='relu',
                       padding='same')(conv4)

        # Conv5 (Output n/16, n/16, 1024)
        conv5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
        conv5 = Conv2D(1024, (3, 3), activation='relu',
                       padding='same')(conv5)
        conv5 = Conv2D(1024, (3, 3), activation='relu',
                       padding='same')(conv5)

        # begin resizing attempt

        # deconv1 (Output n/8, n/8, 512)
        deconv1 = Conv2DTranspose(512, (3, 3), strides=(2, 2),
                                  activation='relu', padding='same')(conv5)
        deconv1_1 = Cropping2D(((0, 0), (1, 1)))(conv4)
        both_1 = Concatenate([deconv1, deconv1_1])

        # deconv2 (Output n/4, n/4, 256)
        print(type(both_1))
        deconv2 = Conv2DTranspose(256, (3, 3), strides=(2, 2),
                                  activation='relu', padding='same')(both_1)
        deconv2_1 = Cropping2D(((0, 0), (1, 1)))(conv3)
        both_2 = Concatenate([deconv2, deconv2_1])

        # deconv3 (Output n/2, n/2, 128)
        deconv3 = Conv2DTranspose(128, (5, 5), strides=(2, 2),
                                  activation='relu', padding='same')(both_2)
        deconv3_1 = Cropping2D(((0, 0), (1, 1)))(conv2)
        both_3 = Concatenate([deconv3, deconv3_1])

        # deconv4 (Output n, n, 96)
        deconv4 = Conv2DTranspose(96, (5, 5), strides=(2, 2),
                                  activation='relu', padding='same')(both_3)
        deconv4_1 = Cropping2D(((0, 0), (1, 1)))(conv1)
        both_4 = Concatenate([deconv4, deconv4_1])

        # Fully connected layers
        dense1 = Dense(1024, activation='relu', use_bias=True)(both_4)
        dense2 = Dense(1024, activation='relu', use_bias=True)(dense1)
        output_img = Dense(num_classes, activation='sigmoid',
                           use_bias=True)(dense2)

        # begin unused code
        # Conv2 (Output 50 x 50 x 64)
        # x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        # # Conv3 (Output 12 x 12 x 96)
        # x = Conv2D(512, (3, 3), activation='relu',
        #            padding='same')(x)

        # # Conv4 (Output 6 x 6 x 128)
        # x = Conv2D(512, (3, 3), activation='relu', strides=(2, 2),
        #            padding='same')(x)

        # # Conv5 (Output 3 x 3 x 128)
        # x = Conv2D(1024, (3, 3), activation='relu', strides=(2, 2),
        #            padding='same')(x)

        # # Flatten
        # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        # x = Flatten()(x)

        # # Fully connected layers
        # x = Dense(1024, activation='relu', use_bias=True)(x)
        # x = Dense(1024, activation='relu', use_bias=True)(x)
        # output_img = Dense(num_classes, activation='sigmoid',
        #                    use_bias=True)(x)

        model = Model(inputs=input_img, outputs=output_img)
        return model


m = SimpleUNetBuilder.build((120, 120, 64))
m.summary()
