"""
This script allows you to train and evaluate models by specifying a
dictionary of data source and hyperparameter values (in args).

Instead of worrying about how to evaluate/debug models, you can
instead just preprocess data, define a model and have this code
take care of the rest.

To use this script, one would configure the arguments at the bottom
of the file and then run the script using nohup on a GPU machine.
You would then come back in a while to see your results (on Slack).

The proposed workflow is:
- define processed data (filepaths)
- define data generators (methods taking in x_train, y_train and other params)
    - define data augmentation as well
- define models to train (methods taking in generators and other params)
    - also define hyperparameters to optimize as well

After doing such, run the script and see your model results on Slack
in a few minutes.

The script assumes that:
- you have ssh access to one of our configured cloud GPU
- you are able to get processed data onto that computer
- you are familiar with Python and the terminal
"""
import datetime
import importlib
import logging
import multiprocessing
import pathlib
import subprocess
import time
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union, Sequence

import keras
import numpy as np
import os
import pandas as pd
from elasticsearch_dsl import connections
from sklearn import model_selection

import blueno
from blueno import (
    utils,
    elasticsearch,
    io,
    preprocessing,
    transforms,
)

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def configure_parent_logger(file_name,
                            stdout=True):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(file_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    if stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)


def configure_job_logger(file_path):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(file_path)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def to_arrays(data: Dict[str, np.ndarray],
              labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the data and labels into numpy arrays.

    Note: This function filters mismatched labels.

    Note: The index of labels must be patient IDs.

    :param data:
    :param labels: a dataframe WITH patient ID for the index.
    :return: three arrays: the arrays, then the labels, then the corresponding
    ids
    """
    patient_ids = data.keys()
    X_list = []
    y_list = []
    remaining_ids = []
    for id_ in patient_ids:
        try:
            y_list += [labels.loc[id_]]
            X_list += [data[id_]]  # Needs to be in this order
            remaining_ids += [id_]
        except KeyError:
            logging.warning(f'{id_} in data was not present in labels')
            logging.warning(f'{len(X_list)}, {len(y_list)}')
    for id_ in labels.index.values:
        if id_ not in patient_ids:
            logging.warning(f'{id_} in labels was not present in data')

    assert len(X_list) == len(y_list)
    assert len(X_list) == len(remaining_ids)
    return np.stack(X_list), np.stack(y_list), np.array(remaining_ids)


RAW_ARRAYS = {}


def preprocess_data(data_dir: str,
                    labels_path: str,
                    height_offset: int,
                    mip_thickness: int,
                    pixel_value_range: Sequence) -> Tuple[dict, pd.DataFrame]:
    """
    Preprocesses data by loading raw numpy data, cleaning, filtering,
    and transforming the arrays and labels.

    It is recommended that you use your own preprocessing callable
    if you want more configurability.

    :param data_dir: path to a directory containing .npz files
    :param labels_path: path a directory containing the 2 metadata CSVs
    :param height_offset: the offset from the top of the 3D image to crop from
    :param mip_thickness: the thickness of a single mip
    :param pixel_value_range: the range of pixel values to crop to
    :return:
    """
    global RAW_ARRAYS
    if RAW_ARRAYS:
        logging.info('loading cached arrays')
        raw_arrays = RAW_ARRAYS
    else:
        logging.info('loading raw arrays from disk')
        raw_arrays = io.load_compressed_arrays(data_dir)
        RAW_ARRAYS = raw_arrays

    raw_labels = io.load_labels(labels_path)

    cleaned_arrays, cleaned_labels = preprocessing.clean_data(raw_arrays,
                                                              raw_labels)

    def filter_data(arrays: dict, labels: pd.DataFrame):
        filtered_arrays = {id_: arr for id_, arr in arrays.items()
                           if arr.shape[0] != 1}  # Remove the bad array
        filtered_labels = labels.loc[filtered_arrays.keys()]
        assert len(filtered_arrays) == len(filtered_labels)
        return filtered_arrays, filtered_labels

    filtered_arrays, filtered_labels = filter_data(cleaned_arrays,
                                                   cleaned_labels)

    def process_data(arrays: dict,
                     labels: pd.DataFrame,
                     height_offset: int,
                     mip_thickness: int,
                     pixel_value_range: Sequence):
        processed_arrays = {}
        for id_, arr in arrays.items():
            try:
                arr = transforms.crop(arr,
                                      (3 * mip_thickness, 200, 200),
                                      height_offset=height_offset)
                arr = np.stack([arr[:mip_thickness],
                                arr[mip_thickness: 2 * mip_thickness],
                                arr[2 * mip_thickness: 3 * mip_thickness]])
                arr = transforms.bound_pixels(arr,
                                              pixel_value_range[0],
                                              pixel_value_range[1])
                arr = arr.max(axis=1)
                arr = arr.transpose((1, 2, 0))
                processed_arrays[id_] = arr
            except AssertionError:
                print(f'patient id {id_} could not be processed,'
                      f' has input shape {arr.shape}')
        processed_labels = labels.loc[
            processed_arrays.keys()]  # Filter, if needed
        assert len(processed_arrays) == len(processed_labels)
        return processed_arrays, processed_labels

    processed_arrays, processed_labels = process_data(
        filtered_arrays,
        filtered_labels,
        height_offset=height_offset,
        mip_thickness=mip_thickness,
        pixel_value_range=pixel_value_range
    )
    return processed_arrays, processed_labels


def prepare_data(params: blueno.ParamConfig) -> Tuple[np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray]:
    """
    Prepares the data referenced in params for ML. This includes
    shuffling and expanding dims.

    :param params: a hyperparameter dictionary generated from a ParamGrid
    :return: x_train, x_valid, y_train, y_valid
    """
    logging.info(f'using params:\n{params}')
    # Load the arrays and labels
    data_params = params.data
    array_dict = io.load_arrays(data_params.data_dir)
    index_col = data_params.index_col
    label_col = data_params.label_col
    label_series = pd.read_csv(data_params.labels_path,
                               index_col=index_col)[label_col]
    # Convert to numpy arrays
    x, y, patient_ids = to_arrays(array_dict, label_series)

    if params.model.loss == keras.losses.categorical_crossentropy:
        y = keras.utils.to_categorical(y)
    elif params.model.loss == keras.losses.binary_crossentropy:
        if y.ndim == 1:
            y = np.expand_dims(y, axis=-1)

    assert y.ndim == 2

    logging.debug(f'x shape: {x.shape}')
    logging.debug(f'y shape: {y.shape}')
    logging.info(f'seeding to {params.seed} before shuffling')

    x_train, x_valid, y_train, y_valid, ids_train, ids_valid = \
        model_selection.train_test_split(
            x, y, patient_ids,
            test_size=params.val_split,
            random_state=params.seed)
    return x_train, x_valid, y_train, y_valid, ids_train, ids_valid


def start_job(x_train: np.ndarray,
              y_train: np.ndarray,
              x_valid: np.ndarray,
              y_valid: np.ndarray,
              job_name: str,
              username: str,
              params: blueno.ParamConfig,
              slack_token: str = None,
              log_dir: str = None,
              id_valid: np.ndarray = None) -> None:
    """
    Builds, fits, and evaluates a model.

    If slack_token is not none, uploads an image.

    For advanced users it is recommended that you input your own job
    function and attach desired loggers.

    :param x_train:
    :param y_train: the training labels, must be a 2D array
    :param x_valid:
    :param y_valid: the validation labels, must be  a 2D array
    :param job_name:
    :param username:
    :param slack_token: the slack token
    :param params: the parameters specified
    :param log_dir:
    :param id_valid: the patient ids ordered to correspond with y_valid
    :return:
    """
    num_classes = y_train.shape[1]
    created_at = datetime.datetime.utcnow().isoformat()

    # Configure the job to log all output to a specific file
    csv_filepath = None
    log_filepath = None
    if log_dir:
        log_filepath = str(
            pathlib.Path(log_dir) / f'{job_name}-{created_at}.log')
        csv_filepath = log_filepath[:-3] + 'csv'
        configure_job_logger(log_filepath)

    # This must be the first lines in the jo log, do not change
    logging.info(f'using params:\n{params}')
    logging.info(f'author: {username}')

    logging.debug(f'in start_job,'
                  f' using gpu {os.environ["CUDA_VISIBLE_DEVICES"]}')

    logging.info('preparing data and model for training')

    model_params = params.model
    generator_params = params.generator

    train_gen, valid_gen = generator_params.generator_callable(
        x_train, y_train,
        x_valid, y_valid,
        params.batch_size,
        **generator_params.__dict__)

    logging.debug(f'num_classes is: {num_classes}')

    # Construct the uncompiled model
    model: keras.Model
    model = model_params.model_callable(input_shape=x_train.shape[1:],
                                        num_classes=num_classes,
                                        **model_params.__dict__)

    logging.debug(
        'using default metrics: acc, sensitivity, specificity, tp, fn')
    metrics = ['acc',
               utils.sensitivity,
               utils.specificity,
               utils.true_positives,
               utils.false_negatives]

    model.compile(optimizer=model_params.optimizer,
                  loss=model_params.loss,
                  metrics=metrics)

    model_filepath = '/tmp/{}.hdf5'.format(os.environ['CUDA_VISIBLE_DEVICES'])
    logging.debug('model_filepath: {}'.format(model_filepath))
    callbacks = utils.create_callbacks(x_train, y_train, x_valid, y_valid,
                                       early_stopping=params.early_stopping,
                                       reduce_lr=params.reduce_lr,
                                       csv_file=csv_filepath,
                                       model_file=model_filepath)
    logging.info('training model')
    history = model.fit_generator(train_gen,
                                  epochs=params.max_epochs,
                                  validation_data=valid_gen,
                                  verbose=2,
                                  callbacks=callbacks)

    if slack_token:
        logging.info('generating slack report')
        blueno.slack.slack_report(x_train, x_valid, y_valid, model, history,
                                  job_name,
                                  params, slack_token, id_valid=id_valid)
    else:
        logging.info('no slack token found, not generating report')

    acc_i = model.metrics_names.index('acc')
    if model.evaluate_generator(valid_gen)[acc_i] >= 0.8:
        upload_model_to_gcs(job_name, created_at, model_filepath)

    end_time = datetime.datetime.utcnow().isoformat()
    # Do not change, this generates the ended at ES field
    logging.info(f'end time: {end_time}')

    # Upload logs to Kibana
    if log_dir:
        # Creates a connection to our Airflow instance
        alias = f'bluenot-{os.environ["CUDA_VISIBLE_DEVICES"]}'
        connections.create_connection(hosts=['http://104.196.51.205'],
                                      alias=alias)
        elasticsearch.insert_or_ignore_filepaths(
            pathlib.Path(log_filepath),
            pathlib.Path(csv_filepath),
            alias=alias,
        )
        connections.remove_connection(alias)


def upload_model_to_gcs(job_name, created_at, model_filepath):
    gcs_filepath = 'gs://elvos/models/{}-{}.hdf5'.format(
        # Remove the extension
        job_name,
        created_at,
    )
    # Do not change, this is log is used to get the gcs link
    logging.info('uploading model {} to {}'.format(
        model_filepath,
        gcs_filepath,
    ))

    try:
        subprocess.run(
            ['/bin/bash',
             '-c',
             'gsutil cp {} {}'.format(model_filepath, gcs_filepath)],
            check=True)
    except subprocess.CalledProcessError:
        # gpu1708 specific code
        subprocess.run(
            ['/bin/bash',
             '-c',
             '/gpfs/main/home/lzhu7/google-cloud-sdk/bin/'
             'gsutil cp {} {}'.format(model_filepath, gcs_filepath)],
            check=True)


def hyperoptimize(hyperparams: Union[blueno.ParamGrid,
                                     List[blueno.ParamConfig]],
                  username: str,
                  slack_token: str = None,
                  num_gpus=1,
                  gpu_offset=0,
                  log_dir: str = None) -> None:
    """
    Runs training jobs on input hyperparameter grid.

    :param hyperparams: a dictionary of parameters. See blueno/types for
    a specification
    :param username: your name
    :param slack_token: a slack token for uploading to GitHub
    :param num_gpus: the number of gpus you will use
    :param gpu_offset: your gpu offset
    :param log_dir: the directory you will too. This directory should already
    exist
    :return:
    """
    if isinstance(hyperparams, blueno.ParamGrid):
        param_list = model_selection.ParameterGrid(hyperparams.__dict__)
    else:
        param_list = hyperparams

    logging.info(
        'optimizing grid with {} configurations'.format(len(param_list)))

    gpu_index = 0
    processes = []
    for params in param_list:
        if isinstance(params, dict):
            params = blueno.ParamConfig(**params)
        # This is where we'd run preprocessing. To run in a reasonable amount
        # of time, the raw data must be cached in-memory.
        x_train, x_valid, y_train, y_valid, id_train, id_valid = prepare_data(
            params)

        # Start the model training job
        # Run in a separate process to avoid memory issues
        # Note how this depends on offset
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_index + gpu_offset}'

        if params.job_fn is None:
            job_fn = start_job
        else:
            job_fn = params.job_fn

        logging.debug('using job fn {}'.format(job_fn))

        # Uses the parent of the data_dir to name the job,
        # which may not work for all data formats.
        job_name = str(pathlib.Path(params.data.data_dir).parent)
        job_name += f'_{y_train.shape[1]}-classes'

        process = multiprocessing.Process(target=job_fn,
                                          args=(x_train, y_train,
                                                x_valid, y_valid),
                                          kwargs={
                                              'params': params,
                                              'job_name': job_name,
                                              'username': username,
                                              'slack_token': slack_token,
                                              'log_dir': log_dir,
                                              'id_valid': id_valid,
                                          })
        gpu_index += 1
        gpu_index %= num_gpus

        logging.debug(f'gpu_index is now {gpu_index + gpu_offset}')
        process.start()
        processes.append(process)
        if gpu_index == 0:
            logging.info(f'all gpus used, calling join on processes:'
                         f' {processes}')
            p: multiprocessing.Process
            for p in processes:
                p.join()
            processes = []
            time.sleep(60)


def check_config(config):
    logging.info('checking that config has all required attributes')
    logging.debug('replace arguments with PARAM_GRID')
    logging.debug('PARAM_GRID: {}'.format(config.PARAM_GRID))
    logging.debug('USER: {}'.format(config.USER))
    logging.debug('NUM_GPUS: {}'.format(config.NUM_GPUS))
    logging.debug('GPU_OFFSET: {}'.format(config.GPU_OFFSET))
    gpu_range = range(config.GPU_OFFSET, config.GPU_OFFSET + config.NUM_GPUS)
    logging.info('using GPUs: {}'.format([x for x in gpu_range]))
    logging.debug('BLUENO_HOME: {}'.format(config.BLUENO_HOME))
    logging.debug('LOG_DIR: {}'.format(config.LOG_DIR))
    logging.debug('SLACK_TOKEN: {}'.format(config.SLACK_TOKEN))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config',
                        help='The config module (ex. config_luke)',
                        default='config-1')
    args = parser.parse_args()

    logging.info('using config {}'.format(args.config))
    user_config = importlib.import_module(args.config)

    parent_log_file = pathlib.Path(
        user_config.LOG_DIR) / 'results-{}.txt'.format(
        datetime.datetime.utcnow().isoformat()
    )
    configure_parent_logger(parent_log_file)
    check_config(user_config)

    logging.info('checking param grid')
    if isinstance(user_config.PARAM_GRID, blueno.ParamGrid):
        param_grid = user_config.PARAM_GRID
    elif isinstance(user_config.PARAM_GRID, list):
        param_grid = user_config.PARAM_GRID
    elif isinstance(user_config.PARAM_GRID, dict):
        logging.warning('creating param grid from dictionary, it is'
                        'recommended that you define your config'
                        'with ParamGrid')
        param_grid = blueno.ParamGrid(**user_config.PARAM_GRID)
    else:
        raise ValueError('user_config.PARAM_GRID must be a ParamGrid,'
                         ' list, or dict')
    hyperoptimize(param_grid,
                  user_config.USER,
                  user_config.SLACK_TOKEN,
                  num_gpus=user_config.NUM_GPUS,
                  gpu_offset=user_config.GPU_OFFSET,
                  log_dir=user_config.LOG_DIR)
