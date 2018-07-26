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

from keras.layers import Input
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model


class Cad3dBuilder(object):

    @staticmethod
    def build(input_shape, filters=(64, 128, 256),
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

        if len(filters) < 1:
            raise ValueError("Filter layers need to be more than 1")

        input_img = Input(shape=input_shape, name="cad_input")
        x = input_img

        num_encoding_layers = len(filters)

        for i in range(num_encoding_layers):
            x = Conv3D(filters[i], filter_size,
                       activation='relu', padding='same',
                       name="cad_enc_{}".format(i))(x)
            x = MaxPooling3D(pool_size=pool_size, strides=(2, 2, 2),
                             padding="same",
                             name="cad_pool_{}".format(i))(x)

        for i in range(num_encoding_layers)[::-1]:
            x = Conv3D(filters[i], filter_size,
                       activation='relu', padding='same',
                       name="cad_dec_{}".format(i))(x)
            x = UpSampling3D(size=pool_size, data_format=None,
                             name="cad_unpool_{}".format(i))(x)

        x = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same',
                   name="cad_sigmoid")(x)
        model = Model(inputs=input_img, outputs=x)
        return model

# m = Cad3dBuilder.build((200, 200, 200, 1), filters=(8, 8, 8))
# m.summary()
