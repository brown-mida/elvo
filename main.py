from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from generators.mnist_generator import MnistGenerator
from generators.alexnet_generator import AlexNetGenerator
from generators.new_generator import NewGenerator
from generators.single_generator import SingleGenerator

from models.alexnet3d import AlexNet3DBuilder
from models.alexnet2d import AlexNet2DBuilder
from models.simple import SimpleNetBuilder


def train_simplenet():
    # Parameters
    dim_len = 120
    top_len = 64
    epochs = 10
    batch_size = 4

    # Generators
    training_gen = NewGenerator(
        batch_size=batch_size,
        augment_data=False,
        extend_dims=False
    )
    validation_gen = NewGenerator(
        batch_size=batch_size,
        augment_data=False,
        extend_dims=False,
        validation=True
    )

    # Build and run model
    model = SimpleNetBuilder.build((dim_len, dim_len, top_len))
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy',
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


def train_alexnet2d():
    # Parameters
    dim_len = 120
    top_len = 64
    epochs = 10
    batch_size = 4

    # Generators
    training_gen = NewGenerator(
        batch_size=batch_size,
        split=0.1,
        augment_data=True,
        extend_dims=False
    )
    validation_gen = NewGenerator(
        batch_size=batch_size,
        extend_dims=False,
        split=0.1,
        augment_data=True,
        validation=True
    )

    # Build and run model
    model = AlexNet2DBuilder.build((dim_len, dim_len, top_len))
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy',
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
        max_queue_size=5)
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
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy',
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
    train_simplenet()
