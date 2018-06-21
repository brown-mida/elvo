"""
A simple 3D Convolutional network. It has 5 convolutional
layers and 3 FC layers, much like Alexnet.

Unfortunately, the architecture described in the paper is
severely lacking in details. Here I use 2D convolution
(with depth as channels) instead of 3D. In addition, I assume:
- The filter sizes
- The strides between convolutional layers
- The pooling size

Based on:
- https://github.com/ehosseiniasl/3d-convolutional-network
- https://www.nature.com/articles/s41746-017-0015-z.pdf
"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.layers.convolutional import Conv2D, MaxPooling2D

# from ml.models.model import ModelBuilder


class AlexNet2DBuilder(object):

    def __add_new_last_layer(self, base_model, nb_classes):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(input=base_model.input, output=predictions)
        return model

    @staticmethod
    def build(input_shape, num_classes=2):
        base_model = ResNet50(weights='imagenet', include_top=False)

# m = AlexNet2DBuilder.build((120, 120, 64))
# m.summary()
