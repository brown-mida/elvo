from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from generators.alexnet_generator import AlexNetGenerator
from models.alexnet3d import AlexNet3DBuilder


def train_alexnet3d():
    # Parameters
    dim_len = 200
    top_len = 24
    epochs = 10
    batch_size = 4

    # Generators
    training_gen = AlexNetGenerator(
        dims=(dim_len, dim_len, top_len),
        batch_size=batch_size,
    )
    validation_gen = AlexNetGenerator(
        dims=(dim_len, dim_len, top_len),
        batch_size=batch_size,
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
