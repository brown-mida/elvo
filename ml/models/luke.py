import logging

import keras


def resnet(input_shape=(224, 224, 3), num_classes=1, dropout_rate1=0.5,
           dropout_rate2=0.5, freeze=False, **kwargs) -> keras.models.Model:
    """Returns a uncompiled, pretrained ResNet50.
    :param freeze:
    """
    resnet = keras.applications.ResNet50(include_top=False,
                                         input_shape=input_shape)

    if freeze:
        layer: keras.layers.Layer
        for layer in resnet.layers:
            layer.trainable = False

    predictions = top_model(resnet.output, num_classes, dropout_rate1,
                            dropout_rate2)

    model = keras.models.Model(resnet.input, predictions)
    return model


def inception(input_shape=(224, 224, 3), num_classes=1, dropout_rate1=0.5,
              dropout_rate2=0.5, freeze=False, **kwargs) -> keras.models.Model:
    """Returns a uncompiled, inceptionV3 model.
    :param freeze:
    """
    inception = keras.applications.InceptionV3(include_top=False,
                                               input_shape=input_shape)
    predictions = top_model(inception.output, num_classes, dropout_rate1,
                            dropout_rate2)

    if freeze:
        layer: keras.layers.Layer
        for layer in inception.layers:
            layer.trainable = False

    model = keras.models.Model(inception.input, predictions)
    return model


def inception_resnet(input_shape=(224, 224, 3),
                     num_classes=1,
                     dropout_rate1=0.5,
                     dropout_rate2=0.5,
                     freeze=False,
                     **kwargs) -> keras.models.Model:
    """Returns a uncompiled, inception-resnet model.
    """
    inception = keras.applications.InceptionResNetV2(
        include_top=False, input_shape=input_shape)

    if freeze:
        layer: keras.layers.Layer
        for layer in inception.layers:
            layer.trainable = False

    predictions = top_model(inception.output,
                            num_classes, dropout_rate1,
                            dropout_rate2)

    model = keras.models.Model(inception.input, predictions)
    return model


def nasnet(input_shape=(224, 224, 3),
           num_classes=1,
           dropout_rate1=0.5,
           dropout_rate2=0.5,
           freeze=False,
           **kwargs):
    nasnet = keras.applications.NASNetMobile(
        include_top=False, input_shape=input_shape)

    if freeze:
        layer: keras.layers.Layer
        for layer in nasnet.layers:
            layer.trainable = False

    predictions = top_model(nasnet.output, num_classes, dropout_rate1,
                            dropout_rate2)

    model = keras.models.Model(nasnet.input, predictions)
    return model


def top_model(pretrained: keras.layers.Layer,
              num_classes,
              dropout_rate1,
              dropout_rate2):
    x = pretrained
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate1)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate2)(x)

    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    logging.debug(f'using activation: {activation}')

    predictions = keras.layers.Dense(num_classes, activation=activation)(x)

    return predictions
