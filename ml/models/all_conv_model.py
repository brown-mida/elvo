from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from ml.generators.mnist_generator import MnistGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class AllConvModelBuilder(object):

    @staticmethod
    def build(input_shape):

        if len(input_shape) != 3:
            raise ValueError("Input shape should be a tuple of the form (conv1, conv2, conv3)")

        output_dim = 2
        filt_dims = 3
        dropout = 0.2

        # input shape for mip images is (120, 120, 1)
        input_img = Input(shape=input_shape)

        # Conv1: output shape (120, 120, 96)
        x = Conv2D(96, (filt_dims, filt_dims), activation='relu', padding='same')(input_img)
        x = BatchNormalization()(x)

        # Conv2: output shape (120, 120, 96)
        x = Conv2D(96, (filt_dims, filt_dims), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Conv3: output shape (30, 30, 96)
        x = Conv2D(96, (filt_dims, filt_dims), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        # Conv4: output shape (30, 30, 192)
        x = Conv2D(192, (filt_dims, filt_dims), activation='relu', padding='same')(input_img)
        x = BatchNormalization()(x)

        # Conv5: output shape (30, 30, 192)
        x = Conv2D(192, (filt_dims, filt_dims), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Conv6: output shape (15, 15, 192)
        x = Conv2D(192, (filt_dims, filt_dims), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        # Conv7: output shape (15, 15, 192)
        x = Conv2D(192, (filt_dims, filt_dims), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Conv8: output shape (15, 15, 192)
        x = Conv2D(192, (1, 1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Conv9: output shape (15, 15, 2)
        x = Conv2D(10, (1, 1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D(data_format='channels_last')(x)

        output_img = Dense(10, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=output_img)
        return model


if __name__ == '__main__':

    batch_size = 32
    dim_len = 120
    channels = 1
    epochs = 10

    Gen = MnistGenerator

    training_gen = Gen(
        dims=(120, 120, 1),
        batch_size=batch_size,
        augment_data=False,
        extend_dims=False
    )
    validation_gen = Gen(
        dims=(120, 120, 1),
        batch_size=batch_size,
        augment_data=False,
        extend_dims=False,
        validation=True
    )

    m = AllConvModelBuilder.build((120, 120, 1))
    m.summary()
    m.compile(optimizer=Adam(lr=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    mc_callback = ModelCheckpoint(filepath='tmp/alex_weights.hdf5', verbose=1)
    print('Model has been compiled.')

    # Training
    m.fit_generator(
        generator=training_gen.generate(),
        steps_per_epoch=training_gen.get_steps_per_epoch(),
        validation_data=validation_gen.generate(),
        validation_steps=validation_gen.get_steps_per_epoch(),
        epochs=epochs,
        callbacks=[mc_callback],
        verbose=1,
        max_queue_size=1)
    print('Model has been fit.')

