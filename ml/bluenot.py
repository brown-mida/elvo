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

# TODO: Better model evaluation:
- more plots generation (PR curve, t-SNE, etc.)

# TODO: Incorporate preprocessing
- hyperparameter optimization on crop, etc.


TODO: from Michelangelo:
Who trained the model
Start and end time of the training job
Full model configuration (features used, hyper-parameter values, etc.)
Reference to training and test data sets
Distribution and relative importance of each feature
Model accuracy metrics
Standard charts and graphs for each model type
(e.g. ROC curve, PR curve, and confusion matrix for a binary classifier)
Full learned parameters of the model
Summary statistics for model visualization
"""
import contextlib
import datetime
import logging
import multiprocessing
import pathlib
import time
import typing

import os

# So that the parent process cannot access gpus
import config

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import os
import pandas as pd
from sklearn import model_selection

import utils


def load_arrays(data_dir: str) -> typing.Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def to_arrays(data: typing.Dict[str, np.ndarray],
              labels: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
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


def start_job(x_train: np.ndarray,
              y_train: np.ndarray,
              x_valid: np.ndarray,
              y_valid: np.ndarray,
              name: str = None,
              params: dict = None) -> None:
    """Builds, fits, and evaluates a model.

    Uploads a report to Slack at the end
    """
    if name is None:
        name = params['data']['data_dir'].split('/')[-3]

    # Configure the job to log all output to a specific file
    isostr = datetime.datetime.utcnow().isoformat()
    log_filename = f'{config.BLUENO_HOME}logs/{name}-{isostr}.log'
    config.configure_job_logger(log_filename)
    contextlib.redirect_stdout(log_filename)
    contextlib.redirect_stderr(log_filename)

    logging.debug(f'in start_job,'
                  f' using gpu {os.environ["CUDA_VISIBLE_DEVICES"]}')

    logging.info('preparing data and model for training')
    model_params = params['model']

    train_gen, valid_gen = params['generator'](x_train, y_train,
                                               x_valid, y_valid,
                                               model_params['rotation_range'],
                                               model_params['batch_size'])

    # Construct the uncompiled model
    model = model_params['model_callable'](input_shape=x_train.shape[1:],
                                           **model_params)

    logging.debug(
        'using default metrics: acc, sensitivity, specificity, tp, fn')
    metrics = ['acc',
               utils.sensitivity,
               utils.specificity,
               utils.true_positives,
               utils.false_negatives]

    model.compile(optimizer=model_params['optimizer'],
                  loss=model_params['loss'],
                  metrics=metrics)

    csv_filename = log_filename[:-3] + 'csv'
    callbacks = utils.create_callbacks(x_train, y_train, x_valid, y_valid,
                                       csv_filename)
    logging.info('training model')
    history = model.fit_generator(train_gen,
                                  epochs=100,
                                  validation_data=valid_gen,
                                  verbose=2,
                                  callbacks=callbacks)

    logging.info('generating slack report')
    utils.slack_report(x_train, y_train,
                       x_valid, y_valid,
                       model, history,
                       name, model_params)


def hyperoptimize(hyperparams: dict) -> None:
    param_grid = model_selection.ParameterGrid(hyperparams)

    gpu_index = 0

    processes = []
    for params in param_grid:
        logging.info(f'using params:\n{params}')

        # Load the arrays and labels
        data_params = params['data']
        array_dict = load_arrays(data_params['data_dir'])
        index_col = data_params['index_col']
        label_col = data_params['label_col']
        label_df = pd.read_csv(data_params['labels_path'],
                               index_col=index_col)[[label_col]]

        # Convert to split numpy arrays
        x, y = to_arrays(array_dict, label_df)

        logging.info(f'seeding to {params["seed"]} before shuffling')
        x_train, x_valid, y_train, y_valid = \
            model_selection.train_test_split(
                x, y,
                test_size=params['val_split'],
                random_state=params["seed"])

        logging.debug(f'training positives: {y_train.sum()}')
        logging.debug(f'training negatives:'
                      f' {len(y_train) - y_train.sum()}')
        logging.debug(f'validation positives:'
                      f' {y_valid.sum()}')
        logging.debug(f'validation negatives:'
                      f' {len(y_valid) - y_valid.sum()}')
        logging.debug(f'x_train mean: {x_train.mean()}')
        logging.debug(f'x_train std: {x_train.std()}')

        # Start the model training job
        # Run in a separate process to avoid memory issues
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_index}'
        process = multiprocessing.Process(target=start_job,
                                          args=(x_train, y_train,
                                                x_valid, y_valid),
                                          kwargs={
                                              'params': params,
                                          })
        gpu_index += 1
        gpu_index %= config.NUM_GPUS
        logging.debug(f'gpu_index is now {gpu_index}')
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


if __name__ == '__main__':
    config.configure_parent_logger()
    hyperoptimize(config.arguments)
