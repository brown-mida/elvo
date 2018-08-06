"""
A simple 3D Convolutional network. It has 5 convolutional
layers and 3 FC layers, much like Alexnet.

Unfortunately, the architecture described in the paper is
severely lacking in details. Here, I use full 3D convolutions.
In addition, I assume:
- That the depth of the image is kept constant throughout
  the convolutional layers
- The filter sizes
- The strides between convolutional layers
- The pooling size

Based on:
- https://github.com/ehosseiniasl/3d-convolutional-network
- https://www.nature.com/articles/s41746-017-0015-z.pdf
"""

from keras.layers import Input, BatchNormalization, Dense, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.models import Model


class AlexNet3DBuilder(object):

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

        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels)")

        input_img = Input(shape=input_shape, name="cad_input")

        # Conv1 (Output 120 x 120 x 64 x 128)
        x = Conv3D(128, (7, 7, 7), activation='relu',
                   padding='same')(input_img)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

        # Conv2 (Output 30 x 30 x 16 x 256)
        x = Conv3D(256, (5, 5, 5), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

        # Conv3 (Output 7 x 7 x 8 x 256)
        x = Conv3D(256, (3, 3, 3), activation='relu',
                   padding='same')(x)

        # Conv4 (Output 4 x 4 x 4 x 512)
        x = Conv3D(512, (3, 3, 3), activation='relu', strides=(2, 2, 2),
                   padding='same')(x)

        # Conv5 (Output 2 x 2 x 2 x 1024)
        x = Conv3D(1024, (3, 3, 3), activation='relu', strides=(2, 2, 2),
                   padding='same')(x)

        # Pooling and flatten
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)
        x = Flatten()(x)

        # Fully connected layers
        x = Dense(1024, activation='relu', use_bias=True)(x)
        x = Dense(1024, activation='relu', use_bias=True)(x)
        x = Dense(num_classes, activation='sigmoid', use_bias=True)(x)

        model = Model(inputs=input_img, outputs=x)
        return model
