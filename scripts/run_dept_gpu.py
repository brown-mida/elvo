"""Loads data from GCS"""
import sys
from pathlib import Path

# This allows us to import from models and generators
import keras
from keras import layers

root_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(root_dir)


class TrainingGenerator(object):

    def __next__(self):
        pass


class ValidationGenerator(object):

    def __next__(self):
        pass


if __name__ == '__main__':
    """
    Preprocessing:
    1. Split the numpy arrays to train and test data.
    2. Convert the arrays to (L, W, H) shape
    3. Crop the arrays to L, W, H.
    4. Bound the hounsfield units.
    4. Map the pixels to the [0, 1] range
    5. Save the data into directories.
    
    5. Feed a couple samples through the model
    
    Training:
    Implement the generator.
    """
    BATCH_SIZE = 32
    LENGTH, WIDTH, HEIGHT = (150, 150, 64)  # TODO

    model = keras.Sequential()
    model.add(layers.Conv2D(256,
                            (3, 3),
                            activation='relu',
                            input_shape=(LENGTH, WIDTH, HEIGHT)))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Dense(1024))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    training_generator = TrainingGenerator()
    validation_generator = ValidationGenerator()

    model.fit_generator(training_generator,
                        steps_per_epoch=100,
                        epochs=30,
                        validation_data=validation_generator,
                        validation_steps=50)
