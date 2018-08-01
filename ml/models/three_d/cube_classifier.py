from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)


class CubeClassifierBuilder(object):

    @staticmethod
    def build(input_shape=(28, 28, 1), dropout=(0.25, 5), binary=False):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout[0]))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout[1]))
        if binary:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(7, activation='sigmoid'))

        model.summary(line_length=140)
        return model
