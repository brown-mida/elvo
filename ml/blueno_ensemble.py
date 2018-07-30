"""Script for ensembling models.

See the __main__ block for more info on how to use the script.
"""

import time
from typing import List, Callable

import keras
import numpy as np
import os
import sklearn
from dataclasses import dataclass
from google.cloud import storage
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

import blueno
from blueno.io import load_model
from blueno.preprocessing import prepare_data


@dataclass
class ModelInfo:
    blob_name: str
    job_name: str
    created_at: str
    val_split: float
    seed: int
    data_dir: str
    labels_path: str
    best_val_acc: float
    best_val_loss: float


def simple_ensemble(model_blob_names: List[str],
                    data_dir: str,
                    labels_path: str,
                    loss: Callable,
                    seed: int,
                    val_split: float,
                    train_test_val: bool,
                    sort: bool):
    """Creates an ensemble from the list of model_urls.

    DO NOT mix models in compat/ with those in sorted_models/.

    :param model_blob_names: a list of model blob names, like
        compat/
    :param data_dir: the path to the data used by ALL models
    :param labels_path: the path to the labels used by ALL models
    :param loss: a loss function like keras.losses.categorical_crossentropy,
        used by ALL models
    :param seed: the seed of ALL of the models
    :param val_split: the val split of ALL of the models
    :param train_test_val: True if train_test_val split was used on the models
    :param sort: set to True of you are loading from sorted_models/, false
        if loading from compat/
    :return:
    """

    # Set the params variable from the function arguments,
    # we'll need this to load the data as x_train, y_train, ...
    params = blueno.ParamConfig(
        data=blueno.DataConfig(
            # TODO: Generalize to work for all users
            data_dir=data_dir,
            labels_path=labels_path,
            index_col='Anon ID',
            label_col='occlusion_exists',
            gcs_url='',
        ),
        generator=None,
        model=blueno.ModelConfig(
            model_callable=None,
            optimizer=None,
            # TODO: Some may use a different loss
            loss=loss,
        ),
        batch_size=None,
        seed=seed,
        val_split=val_split
    )

    x_train, x_valid, y_train, y_valid, _, _ = prepare_data(params,
                                                            train_test_val=train_test_val,
                                                            sort=sort)
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True)
    datagen.fit(x_train)

    client = storage.Client(project='elvo-198322')
    bucket = storage.Bucket(client, name='elvos')

    # This is a copy of the ensemble_models function, using
    # model names instead.
    models = []
    time1 = time.time()
    for i, blob_name in enumerate(model_blob_names):
        # Here we load and evaluate each individual model
        # so we can be sure that our data, validation split, and seed
        # are correct
        blob = bucket.get_blob(blob_name)
        if blob is None:
            raise ValueError(f'Blob {blob_name} does not exist')
        model_filepath = f'{i}.hdf5'

        print(f'downloading model {blob_name}')
        time2 = time.time()
        blob.download_to_filename(model_filepath)
        time3 = time.time()
        print(f'seconds to download: {time3 - time2}')

        print(f'loading model {blob_name}')
        model: keras.Model
        model = load_model(model_filepath, compile=True)
        os.remove(model_filepath)
        time4 = time.time()
        print(f'seconds to load: {time4 - time3}')
        # Used to check the model
        evaluate_model(model, datagen, x_valid, y_valid)

        model.name = f'model_{i}'
        models.append(model)

    # Finally we ensemble and evaluate the models here
    print('using models {}'.format(models))
    model_input = layers.Input(shape=models[0].input_shape[1:])
    ensemble = ensemble_models(models, model_input)

    evaluate_model(ensemble, datagen, x_valid, y_valid)
    time7 = time.time()
    print(f'seconds per ensemble: {time7 - time1}', flush=True)


def ensemble_models(models: List[keras.Model],
                    model_input: keras.layers.Input) -> keras.Model:
    """Take in a list of models and a model input and return
    a single ensemble that takes the average of the last layer.

    It also compiles the model as well
    """
    # collect outputs of models in a list
    y_models = [model(model_input) for model in models]
    # averaging outputs
    y_avg = layers.average(y_models)
    # build model from same input and avg output
    model_ens = keras.Model(inputs=model_input,
                            outputs=y_avg,
                            name='ensemble')

    model_ens.compile(loss=models[0].loss,
                      optimizer=models[0].optimizer,
                      metrics=models[0].metrics)

    return model_ens


def evaluate_model(model: keras.Model, datagen, x_valid, y_valid):
    """
    Evaluates the model on x_valid, y_valid.

    :param model:
    :param datagen:
    :param x_valid:
    :param y_valid:
    :return:
    """
    print('evaluating model', model.name)
    time6 = time.time()
    print('metrics:')
    labels = ['loss', 'acc', 'sens', 'spec', 'fp', 'tn']
    print('labels:', labels)
    values = model.evaluate_generator(
        datagen.flow(x_valid, y_valid, batch_size=8))
    print('values:', values)
    # Note that we standardize when we predict x_valid
    x_valid: np.ndarray
    y_pred = model.predict(datagen.standardize(x_valid.astype(np.float32)))
    print('auc score:', sklearn.metrics.roc_auc_score(y_valid, y_pred))
    time7 = time.time()
    print(f'seconds to evaluate: {time7 - time6}')
    return values


if __name__ == '__main__':
    """This script allows you to build ensembles by mainually defining
    model names.
    
    See the docstring for simple_ensemble for more info on how this
    script should be used. Be very careful to get all of the arguments
    correct, otherwise you'll likely get unreasonably good results.
    """
    data_dir = '/gpfs/main/home/lzhu7/elvo-analysis/data/processed-lower' \
               '/arrays'
    labels_path = '/gpfs/main/home/lzhu7/elvo-analysis/data/' \
                  'processed-lower/labels.csv'

    models = [
        'compat/models/processed-lower_2-classes-2018-07-13T09:59:22.804773'
        '.hdf5',
        # 'compat/models/processed-lower_2-classes-2018-07-13T10:34:07
        # .792655.hdf5',
        # 'compat/models/processed-lower_2-classes-2018-07-13T11:44:00
        # .484273.hdf5',
        'compat/models/processed-lower_2-classes-2018-07-13T12:36:44.036950'
        '.hdf5']

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    simple_ensemble(model_blob_names=models,
                    data_dir=data_dir, labels_path=labels_path,
                    loss=keras.losses.categorical_crossentropy,
                    seed=0,
                    val_split=0.1,
                    train_test_val=False,
                    sort=False)
