"""
Evaluation script, to run trained models on testing data.

This script requires a config file that has EVAL_PARAMS,
which is an EvalConfig object. Use the same data, val_split
and seed as the one used to train the model; otherwise,
the results are not guaranteed to be accurate.

TODO:
- Test this file
- Configure the logger to store result data, and upload to some platform
- Configure the script to run multiple models at once
"""

import argparse
import datetime
import importlib
import logging
import pathlib
import sys

import numpy as np

from blueno import preprocessing, utils, types, logger


def evaluate_model(x_test, y_test, model,
                   normalize=True, x_train=None):
    if normalize:
        if x_train is None:
            raise ValueError(('Must specify training data if normalize '
                              'is set to True.'))
        else:
            x_mean = np.array([x_train[:, :, :, 0].mean(),
                               x_train[:, :, :, 1].mean(),
                               x_train[:, :, :, 2].mean()])
            x_std = np.array([x_train[:, :, :, 0].std(),
                              x_train[:, :, :, 1].std(),
                              x_train[:, :, :, 2].std()])
            x_test = (x_test - x_mean) / x_std

    results = model.evaluate(x=x_test, y=y_test, batch_size=1,
                             verbose=1)
    return results


def evaluate_from_config(params):
    # Load the data
    logging.info('Preparing data and models')
    (x_train, _, x_test, y_train, _, y_test,
     id_train, _, id_test) = preprocessing.prepare_data(params,
                                                        train_test_val=True)

    metrics = ['acc',
               utils.sensitivity,
               utils.specificity,
               utils.true_positives,
               utils.false_negatives]

    model = params.model.model_callable
    model.load_weights(params.model_weights)
    model.compile(loss=params.model.loss, optimizer=params.model.optimizer,
                  metrics=metrics)

    logging.info('Evaluating the model...')
    results = evaluate_model(x_test, y_test, model, x_train=x_train)
    logging.info('Results:')
    logging.info(results)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Evaluation script.')
    parser.add_argument('--config',
                        help='The config module (ex. config_luke)',
                        default='config-1')
    return parser.parse_args(args)


def check_config(config):
    logging.info('Checking that config has all required attributes')
    logging.debug('EVAL_PARAMS: {}'.format(config.EVAL_PARAMS))


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    throw
    logging.info('Using config {}'.format(args.config))
    user_config = importlib.import_module(args.config)

    parent_log_file = pathlib.Path(
        user_config.LOG_DIR) / 'results-{}.txt'.format(
        datetime.datetime.utcnow().isoformat()
    )
    logger.configure_parent_logger(parent_log_file)
    check_config(user_config)

    if not isinstance(user_config.EVAL_PARAMS, types.EvalConfig):
        raise ValueError(('user_config.EVAL_PARAMS must '
                          'be an EvalConfig object'))

    evaluate_from_config(user_config.EVAL_PARAMS)
