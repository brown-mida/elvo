import itertools
import multiprocessing
import pathlib
import time
from typing import List, Tuple

import elasticsearch_dsl
import keras
import numpy as np
import os
import random
from dataclasses import dataclass
from elasticsearch_dsl import connections
from elasticsearch_dsl.response import Hit
from google.cloud import storage
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

import blueno
from blueno import utils
from blueno.io import load_model
from bluenot import prepare_data


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


def check_models(limit=10):
    client = storage.Client(project='elvo-198322')
    bucket = storage.Bucket(client, name='elvos')

    connections.create_connection(hosts=['http://104.196.51.205'])

    models = model_infos(bucket)
    print(models)
    for m in models[:limit]:
        p = multiprocessing.Process(target=load_and_evaluate,
                                    args=(m, bucket))
        p.start()
        p.join()


def load_and_evaluate(model_info: ModelInfo, bucket):
    print('loading and evaluating', model_info, flush=True)
    if '1-classes' in model_info.job_name:
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.categorical_crossentropy

    params = blueno.ParamConfig(
        data=blueno.DataConfig(
            # TODO: Generalize to work for all users
            data_dir=model_info.data_dir.replace('lzhu7', 'lukezhu'),
            labels_path=model_info.data_dir.replace('lzhu7', 'lukezhu'),
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
        seed=model_info.seed,
        val_split=model_info.val_split
    )

    x_train, x_valid, y_train, y_valid, _, _ = prepare_data(params)
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True)
    datagen.fit(x_train)

    model_filepath = f'load_and_eval.hdf5'
    blob = bucket.get_blob(model_info.blob_name)
    blob.download_to_filename(model_filepath)
    model: keras.Model
    model = load_model(model_filepath, compile=True)
    os.remove(model_filepath)

    metrics = evaluate_model(datagen, model, x_valid, y_valid)
    print('actual best loss:', metrics[0])
    print('logged best loss:', model_info.best_val_loss)
    print('actual best acc:', metrics[1])
    print('logged best acc:', model_info.best_val_acc, flush=True)


def ensemble(seed=0,
             val_split=0.1,
             min_best_val_acc=0.85,
             min_best_val_loss=None,  # TODO
             data_dir=pathlib.Path(
                 '/gpfs/main/home/lzhu7/elvo-analysis/data/'
                 'processed-lower/arrays'),
             labels_path=pathlib.Path(
                 '/gpfs/main/home/lzhu7/elvo-analysis/data/'
                 'processed-lower/labels.csv')):
    """
    A long-running job which outputs the results of
    ensembling different models.

    Requirements:
    - knows the seed + val_split and only does those
    permutations
    - can load models off of gcs
    - can upload plots to Slack

    :return:
    """
    client = storage.Client(project='elvo-198322')
    bucket = storage.Bucket(client, name='elvos')

    connections.create_connection(hosts=['http://104.196.51.205'])

    models = model_infos(bucket)
    models0 = [m for m in models
               if m.val_split == val_split
               and m.seed == seed
               and m.best_val_acc >= min_best_val_acc
               and str(data_dir.parent.name) + '/' in m.data_dir]

    print('matching models', models0)
    print('# of models to try', len(models0),
          flush=True)

    params = blueno.ParamConfig(
        data=blueno.DataConfig(
            data_dir=str(data_dir),
            labels_path=str(labels_path),
            index_col='Anon ID',
            label_col='occlusion_exists',
            gcs_url='',
        ),
        generator=None,
        model=blueno.ModelConfig(
            model_callable=None,
            optimizer=None,
            # TODO: Some may have different
            loss=keras.losses.categorical_crossentropy,
        ),
        batch_size=None,
        seed=seed,
        val_split=val_split
    )

    x_train, x_valid, y_train, y_valid, _, _ = prepare_data(params)

    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True)
    datagen.fit(x_train)

    combinations = list(itertools.combinations(models0, 3))
    random.shuffle(combinations)
    for comb in combinations:
        # Start in a separate process to hopefully
        # combat memory leaks
        p = multiprocessing.Process(
            target=evaluate_ensemble,
            args=(bucket, comb, datagen, x_valid, y_valid)
        )
        p.start()
        p.join()


def evaluate_ensemble(bucket: storage.Bucket,
                      info_combination: List[Tuple],
                      datagen: ImageDataGenerator,
                      x_valid: np.ndarray,
                      y_valid: np.ndarray):
    models = []
    time1 = time.time()
    for i in range(3):
        blob_name = info_combination[i][0]
        blob = bucket.get_blob(blob_name)
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

        evaluate_model(datagen, model, x_valid, y_valid)

        model.name = f'model_{i}'
        models.append(model)

    print('using models {}'.format(models))
    model_input = layers.Input(shape=models[0].input_shape[1:])
    ensemble = ensemble_models(models, model_input)

    print('compiling')
    ensemble.compile(
        optimizer='adam',
        loss=keras.losses.categorical_crossentropy,
        metrics=['acc',
                 utils.sensitivity,
                 utils.specificity,
                 utils.true_positives,
                 utils.false_negatives]
    )
    evaluate_model(datagen, ensemble, x_valid, y_valid)
    time7 = time.time()
    print(f'seconds per ensemble: {time7 - time1}', flush=True)


def evaluate_model(datagen, model: keras.Model, x_valid, y_valid):
    print('evaluating model', model.name)
    time6 = time.time()
    print('metrics:')
    labels = ['loss', 'acc', 'sens', 'spec', 'fp', 'tn']
    print('labels:', labels)
    values = model.evaluate_generator(
        datagen.flow(x_valid, y_valid, batch_size=8))
    print('values:', values)
    time7 = time.time()
    print(f'seconds to evaluate: {time7 - time6}')
    return values


def model_infos(bucket) -> List[ModelInfo]:
    models = []
    blob: storage.Blob
    for blob in bucket.list_blobs(prefix='models/'):
        filename = blob.name.split('/')[-1]
        job_name, created_at = _parse_filename(filename)
        print(job_name, created_at)
        # created_at = dateutil.parser.isoparse(created_at)
        results = (elasticsearch_dsl.Search()
                   .query('match', job_name=job_name)
                   .query('match', created_at=created_at)
                   .execute())
        if len(results.hits) > 1:
            raise ValueError('Hits should not be more than 1')
        hit: Hit
        for hit in results.hits:
            # TODO: Turn the tuple into a dataclass
            model_info = ModelInfo(blob_name=blob.name,
                                   job_name=job_name,
                                   created_at=created_at,
                                   val_split=hit.val_split,
                                   seed=hit.seed,
                                   data_dir=hit.data_dir,
                                   labels_path=hit.labels_path,
                                   best_val_acc=hit.best_val_acc,
                                   best_val_loss=hit.best_val_loss)
            models.append(model_info)
    return models


def _parse_filename(filename: str):
    i = filename.find('201')
    if i == -1:
        raise ValueError(f'Could not find timestamp in filename {filename}')
    name = filename[:i - 1]

    if not filename.endswith('.hdf5'):
        raise ValueError(f'Filename does not end with .hdf5')

    dt = filename[i:-len('.hdf5')]
    return name, dt


def ensemble_models(models, model_input):
    # collect outputs of models in a list
    y_models = [model(model_input) for model in models]
    # averaging outputs
    y_avg = layers.average(y_models)
    # build model from same input and avg output
    model_ens = keras.models.Model(inputs=model_input,
                                   outputs=y_avg,
                                   name='ensemble')

    return model_ens


if __name__ == '__main__':
    # ensemble(
    #     data_dir=pathlib.Path(
    #         '/home/lukezhu/elvo-analysis/data/processed-lower/arrays'),
    #     labels_path=pathlib.Path(
    #         '/home/lukezhu/elvo-analysis/data/processed-lower/labels.csv')
    # )
    check_models()
