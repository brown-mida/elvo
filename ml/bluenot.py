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
import contextlib
import datetime
import importlib
import logging
import multiprocessing
import os
import pathlib
import time
import typing
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection

import blueno
import bluenom
from blueno import utils

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import keras  # noqa: E402


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


def load_arrays(data_dir: str) -> typing.Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def to_arrays(data: typing.Dict[str, np.ndarray],
              labels: pd.Series) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Converts the data and labels into numpy arrays.

    Note: This function filters mismatched labels.

    Note: The index of labels must be patient IDs.

    :param data:
    :param labels: a dataframe WITH patient ID for the index.
    :return: two arrays containing the arrays and the labels
    """
    patient_ids = data.keys()
    X_list = []
    y_list = []
    for id_ in patient_ids:
        try:
            y_list += [labels.loc[id_]]
            X_list += [data[id_]]  # Needs to be in this order
        except KeyError:
            logging.warning(f'{id_} in data was not present in labels')
            logging.warning(f'{len(X_list)}, {len(y_list)}')
    for id_ in labels.index.values:
        if id_ not in patient_ids:
            logging.warning(f'{id_} in labels was not present in data')

    assert len(X_list) == len(y_list)
    return np.stack(X_list), np.stack(y_list)


def prepare_data(params: blueno.ParamConfig) -> typing.Tuple[np.ndarray,
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
    array_dict = load_arrays(data_params.data_dir)
    index_col = data_params.index_col
    label_col = data_params.label_col
    label_series = pd.read_csv(data_params.labels_path,
                               index_col=index_col)[label_col]
    # Convert to split numpy arrays
    x, y = to_arrays(array_dict, label_series)

    if params.model.loss == keras.losses.binary_crossentropy:
        # We need y to have 2 dimensions for the rest of the model
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

    elif params.model.loss == keras.losses.categorical_crossentropy:
        # TODO(#77): Move/refactor hacky code below to bluenot.py
        def categorize(label):
            if any([x in label.lower() for x in ['m1', 'm2', 'mca']]):
                return 2  # mca
            if 'nan' in str(label):
                return 0
            else:
                return 1  # other

        try:
            y = np.array([categorize(label) for label in y])
            label_encoder = sklearn.preprocessing.LabelEncoder()
            y = label_encoder.fit_transform(y)
            logging.debug(
                f'label encoder classes: {label_encoder.classes_}')
        except Exception:
            pass

        y = y.reshape(-1, 1)
        one_hot = sklearn.preprocessing.OneHotEncoder(sparse=False)
        y: np.ndarray = one_hot.fit_transform(y)

    else:
        raise ValueError('Only support for crossentry callables at the moment')

    assert y.ndim == 2

    logging.debug(f'x shape: {x.shape}')
    logging.debug(f'y shape: {y.shape}')
    logging.info(f'seeding to {params.seed} before shuffling')

    x_train, x_valid, y_train, y_valid = \
        model_selection.train_test_split(
            x, y,
            test_size=params.val_split,
            random_state=params.seed)
    return x_train, x_valid, y_train, y_valid


def start_job(x_train: np.ndarray,
              y_train: np.ndarray,
              x_valid: np.ndarray,
              y_valid: np.ndarray,
              job_name: str,
              username: str,
              params: blueno.ParamConfig,
              slack_token: str = None,
              epochs=100,  # deprecated
              log_dir: str = None) -> None:
    """
    Builds, fits, and evaluates a model.

    If slack_token is not none, uploads an image

    :param x_train:
    :param y_train: the training labels, must be a 2D array
    :param x_valid:
    :param y_valid: the validation labels, must be  a 2D array
    :param job_name:
    :param username:
    :param slack_token: the slack token
    :param params: the parameters specified
    :param epochs:
    :param log_dir:
    :return:
    """
    num_classes = y_train.shape[1]

    # Configure the job to log all output to a specific file
    csv_filepath = None
    log_filepath = None
    if log_dir:
        isostr = datetime.datetime.utcnow().isoformat()
        log_filepath = str(pathlib.Path(log_dir) / f'{job_name}-{isostr}.log')
        csv_filepath = log_filepath[:-3] + 'csv'
        configure_job_logger(log_filepath)
        contextlib.redirect_stdout(log_filepath)
        contextlib.redirect_stderr(log_filepath)
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

    callbacks = utils.create_callbacks(x_train, y_train, x_valid, y_valid,
                                       filepath=csv_filepath)
    logging.info('training model')
    history = model.fit_generator(train_gen,
                                  epochs=epochs,
                                  validation_data=valid_gen,
                                  verbose=2,
                                  callbacks=callbacks)

    if slack_token:
        logging.info('generating slack report')
        utils.slack_report(x_train, y_train,
                           x_valid, y_valid,
                           model, history,
                           job_name, params,
                           slack_token)
    else:
        logging.info('no slack token found, not generating report')

    end_time = datetime.datetime.utcnow().isoformat()
    logging.info(f'end time: {end_time}')

    # Upload to Kibana
    if log_dir:
        bluenom.insert_job_by_filepaths(pathlib.Path(log_filepath),
                                        pathlib.Path(csv_filepath))


def hyperoptimize(hyperparams: blueno.ParamGrid,
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
    # TODO (#70)
    param_list = model_selection.ParameterGrid(hyperparams.__dict__)

    gpu_index = 0
    processes = []
    for params in param_list:
        params = blueno.ParamConfig(**params)
        x_train, x_valid, y_train, y_valid = prepare_data(params)

        # Start the model training job
        # Run in a separate process to avoid memory issues
        # Note how this depends on offset
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_index + gpu_offset}'

        if params.job_fn is None:
            job_fn = start_job
        else:
            job_fn = params.job_fn

        logging.info('using job fn {}'.format(job_fn))

        job_name = params.data.data_dir.split('/')[-3]
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
                        default='config')
    args = parser.parse_args()

    user_config = importlib.import_module(args.config)

    parent_log_file = pathlib.Path(
        user_config.LOG_DIR) / 'results-{}.txt'.format(
        datetime.datetime.utcnow().isoformat()
    )
    configure_parent_logger(parent_log_file)
    check_config(user_config)

    logging.info('checking param grid')
    param_grid = blueno.ParamGrid(**user_config.PARAM_GRID)
    logging.info('entire param grid: {}'.format(param_grid))

    hyperoptimize(param_grid,
                  user_config.USER,
                  user_config.SLACK_TOKEN,
                  num_gpus=user_config.NUM_GPUS,
                  gpu_offset=user_config.GPU_OFFSET,
                  log_dir=user_config.LOG_DIR)
