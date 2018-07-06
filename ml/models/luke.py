import keras


def resnet(input_shape=(224, 224, 3),
           dropout_rate1=0.5,
           dropout_rate2=0.5) -> keras.models.Model:
    """Returns a uncompiled, pretrained ResNet50.
    """
    resnet = keras.applications.ResNet50(include_top=False,
                                         input_shape=input_shape)
    x = resnet.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate1)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate2)(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(resnet.input, predictions)
    return model


def inception(input_shape=(224, 224, 3),
              dropout_rate1=0.5,
              dropout_rate2=0.5) -> keras.models.Model:
    """Returns a uncompiled, inceptionV3 model.
    """
    inception = keras.applications.InceptionV3(include_top=False,
                                               input_shape=input_shape)
    x = inception.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate1)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate2)(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inception.input, predictions)
    return model


def inception_resnet(input_shape=(224, 224, 3),
                     dropout_rate1=0.5,
                     dropout_rate2=0.5) -> keras.models.Model:
    """Returns a uncompiled, inception-resnet model.
    """
    inception = keras.applications.InceptionResNetV2(
        include_top=False, input_shape=input_shape)
    x = inception.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate1)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate2)(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inception.input, predictions)
    return model
