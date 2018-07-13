from keras.layers import Input, Conv3D, Dense, MaxPooling3D, Dropout
from keras.models import Model


class sMRIModelBuilder(object):

    @staticmethod
    def build(input_shape, num_classes=1):

        if len(input_shape) != 4:
            raise ValueError("Input shape must have 4 channels")

        input_img = Input(input_shape)
        print(input_img)
        conv1 = Conv3D(64, kernel_size=(3, 3, 32), activation='relu')(input_img)
        print(conv1)
        maxpool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
        print(maxpool1)
        conv2 = Conv3D(64, kernel_size=(3, 3, 32), activation='relu')(maxpool1)
        print(conv2)
        conv3 = Conv3D(64, 3, 3, 32, activation='relu')(conv2)
        print(conv3)
        maxpool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
        print(maxpool2)
        dropout = Dropout(0.5)(maxpool2)
        dense1 = Dense(2048, activation='relu')(dropout)
        print(dense1)
        output_img = Dense(num_classes, activation='softmax')(dense1)
        print(output_img)

        model = Model(inputs=input_img, outputs=output_img)
        return model


model = sMRIModelBuilder.build((32, 32, 32, 1))
print(model.summary())
