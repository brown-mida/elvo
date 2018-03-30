"""
3D Stacked Convolutional Autoencoder implementation (CAE).
The idea is that we want to reduce the dimensionality of
the input 3D image, while learning all the important features
from the image.

The learned weights from the encoder should be used later
as a "compressor" for the input image, which is then fed to
another CNN. This should allow us to reduce the number of
training data required.

Based on
- https://github.com/ehosseiniasl/3d-convolutional-network
- https://arxiv.org/pdf/1607.00556.pdf
"""

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D


class Cad3dBuilder(object):

    @staticmethod
    def build(input_shape, filters=(64, 128, 256), num_encoding_layers=3,
              filter_size=(3, 3, 3), pool_size=(2, 2, 2)):
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

        if num_encoding_layers < 1:
            raise ValueError("Encoding layers need to be more than 1")

        if len(filters) != num_encoding_layers:
            raise ValueError("Number of filters need to be the same"
                             "as the number of encoding layers")

        input_img = Input(shape=input_shape)
        x = input_img

        for i in range(num_encoding_layers):
            x = Conv3D(filters[i], filter_size,
                       activation='relu', padding='same')(x)
            x = MaxPooling3D(pool_size=pool_size, strides=(2, 2, 2),
                             padding="same")(x)

        for i in range(num_encoding_layers)[::-1]:
            if i == 0:
                filter_size = (5, 5, 5)
            else:
                filter_size = (3, 3, 3)

            x = Conv3D(filters[i], filter_size,
                       activation='relu', padding='same')(x)
            x = UpSampling3D(size=pool_size, data_format=None)(x)

        x = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
        model = Model(inputs=input_img, outputs=x)
        return model


# m = Cad3dBuilder.build((200, 200, 200, 1), filters=(8, 8, 8))
# m.summary()
