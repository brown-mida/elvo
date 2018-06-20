from ml.models.model import ModelBuilder

from keras.layers import Input, Conv3D, Dense, MaxPooling3D, Dropout
from keras.models import Model


class FusionNet3dBuilder(ModelBuilder):

    @staticmethod
    def build(input_shape, num_classes=1):

        if len(input_shape) != 4:
            raise ValueError("Input shape must have 4 channels")

        input_img = Input(input_shape)
        conv1 = Conv3D(64, kernel_size=(3, 3, 64),
                       activation='relu')(input_img)
        maxpool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
        conv2 = Conv3D(64, kernel_size=(3, 3, 64),
                       activation='relu')(maxpool1)
        conv3 = Conv3D(64, kernel_size=(3, 3, 64), activation='relu')(conv2)
        maxpool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
        dropout = Dropout(0.5)(maxpool2)
        dense1 = Dense(2048, activation='relu')(dropout)
        output_img = Dense(num_classes, activation='softmax')(dense1)

        model = Model(inputs=input_img, outputs=output_img)
        return model
