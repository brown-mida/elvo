from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16


class VGG16Builder(object):

    def __add_new_last_layer(self, base_model, nb_classes):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(input=base_model.input, output=predictions)
        return model

    @staticmethod
    def build(input_shape):
        base_model = VGG16(weights='imagenet', include_top=False)
        finished_model = __add_new_last_layer(base_model, 2)
        return finished_model
