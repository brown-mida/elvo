from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from generators.mnist_generator import MnistGenerator
from generators.alexnet_generator import AlexNetGenerator
from generators.new_generator import NewGenerator

from models.alexnet3d import AlexNet3DBuilder
from models.resnet3d import Resnet3DBuilder


def train_resnet():
    # Parameters
    dim_len = 64
    top_len = 64
    epochs = 10
    batch_size = 16

    # Generators
    training_gen = MnistGenerator(dims=(dim_len, dim_len, top_len),
                                  batch_size=batch_size)
    validation_gen = MnistGenerator(dims=(dim_len, dim_len, top_len),
                                    batch_size=batch_size,
                                    validation=True)

    # Build and run model
    model = Resnet3DBuilder.build_resnet_34((dim_len, dim_len, top_len, 1), 1)
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    mc_callback = ModelCheckpoint(filepath='tmp/weights.hdf5', verbose=1)
    # tb_callback = TensorBoard(write_images=True)

    print('Model has been compiled.')
    model.fit_generator(
        generator=training_gen.generate(),
        steps_per_epoch=training_gen.get_steps_per_epoch(),
        validation_data=validation_gen.generate(),
        validation_steps=validation_gen.get_steps_per_epoch(),
        epochs=epochs,
        callbacks=[mc_callback],
        verbose=1,
        max_queue_size=1)
    print('Model has been fit.')


def train_alexnet3d():
    # Parameters
    dim_len = 120
    top_len = 64
    epochs = 10
    batch_size = 4

    # Generators
    training_gen = NewGenerator(
        dims=(dim_len, dim_len, top_len),
        batch_size=batch_size,
        augment_data=False
    )
    validation_gen = NewGenerator(
        dims=(dim_len, dim_len, top_len),
        batch_size=batch_size,
        augment_data=False,
        validation=True
    )

    # Build and run model
    model = AlexNet3DBuilder.build((dim_len, dim_len, top_len, 1))
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    mc_callback = ModelCheckpoint(filepath='tmp/alex_weights.hdf5', verbose=1)

    print('Model has been compiled.')
    model.fit_generator(
        generator=training_gen.generate(),
        steps_per_epoch=training_gen.get_steps_per_epoch(),
        validation_data=validation_gen.generate(),
        validation_steps=validation_gen.get_steps_per_epoch(),
        epochs=epochs,
        callbacks=[mc_callback],
        verbose=1,
        max_queue_size=1)
    print('Model has been fit.')


if __name__ == '__main__':
    train_alexnet3d()
