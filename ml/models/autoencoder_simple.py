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


class SimpleCad3DBuilder(object):

    @staticmethod
    def build(input_shape):
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

        # Encoding
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same',
                   name="cad_enc_1")(input_img)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding="same", name="cad_pool_1")(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same',
                   name="cad_enc_2")(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding="same", name="cad_pool_2")(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same',
                   name="cad_enc_3")(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding="same", name="cad_pool_3")(x)

        # Decoding
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same',
                   name="cad_dec_1")(x)
        x = UpSampling3D(size=(2, 2, 2), name="cad_unpool_1")(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same',
                   name="cad_dec_2")(x)
        x = UpSampling3D(size=(2, 2, 2), name="cad_unpool_2")(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same',
                   name="cad_dec_3")(x)
        x = UpSampling3D(size=(2, 2, 2), name="cad_unpool_3")(x)

        output_img = Conv3D(1, (3, 3, 3), activation='sigmoid',
                            padding='same', name="cad_sigmoid")(x)
        model = Model(inputs=input_img, outputs=output_img)
        return model


# m = SimpleCad3DBuilder.build((200, 200, 200, 1))
# m.summary()
