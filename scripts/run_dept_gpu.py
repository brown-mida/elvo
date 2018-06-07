"""Loads data from GCS"""
import sys
from pathlib import Path

# This allows us to import from models and generators
from keras import Sequential
from keras.layers import Conv3D, Flatten, Dense, MaxPool3D

root_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(root_dir)


class MyGenerator:
    pass


if __name__ == '__main__':
    BATCH_SIZE = 32
    model = Sequential()
    # Input shape ??
    model.add(Conv3D(filters=32,
                     kernel_size=5,
                     strides=(2, 2, 2),
                     activation='relu',
                     input_shape=(BATCH_SIZE, 128, 128, 32, 1)))
    model.add(MaxPool3D())
    model.add(Conv3D(filters=64,
                     kernel_size=5,
                     strides=(2, 2, 2),
                     activation='relu'))
    model.add(MaxPool3D())
    model.add(Conv3D(filters=64,
                     kernel_size=5,
                     strides=(2, 2, 2),
                     activation='relu'))
    model.add(MaxPool3D())
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit()
