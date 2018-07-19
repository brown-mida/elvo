import os
import sys
import logging
import argparse
import importlib
import pathlib
import datetime
import random

import numpy as np
import keras
from keras.utils import multi_gpu_model

from blueno.elasticsearch import search_top_models
from blueno import types, preprocessing, utils, logger, slack
from models.luke import resnet
from generators.luke import standard_generators


def get_models_to_train(address, lower, upper, data_dir):
    docs = search_top_models(address, lower=lower, upper=upper)
    docs_to_train = []
    for doc in docs:
        data = {}
        data['purported_accuracy'] = doc.best_val_acc
        data['purported_loss'] = doc.best_val_loss
        data['purported_sensitivity'] = doc.best_val_sensitivity
        data['date'] = doc.created_at
        data['job_name'] = doc.job_name

        # Dataset information
        if 'gcs_url' in doc:
            gcs_url = doc.gcs_url
        else:
            folder = doc.data_dir.split('/')[-2]
            # data_dir naming structure is super inconsistent.
            # Here are some edge cases to cover weird names.
            if folder == 'processed':
                folder = 'processed-sumera-1'

            # If not preprocessed (using numpy_compressed), skip
            if folder == 'numpy':
                continue

            gcs_url = 'gs://elvos/processed/{}'.format(folder)

        folder = os.path.join(data_dir, gcs_url.split('/')[-1])
        data['local_dir'] = folder
        data_info = {
            'data_dir': os.path.join(folder,
                                     'arrays/'),
            'labels_path': os.path.join(folder,
                                        'labels.csv'),
            'index_col': 'Anon ID',
            'label_col': 'occlusion_exists',
            'gcs_url': gcs_url
        }

        # Model information
        model = {
            'dropout_rate1': doc.dropout_rate1,
            'dropout_rate2': doc.dropout_rate2,
            'optimizer': keras.optimizers.Adam(lr=1e-5),
            'loss': keras.losses.categorical_crossentropy,
            'freeze': False,
            'model_callable': resnet,
        }

        # Generator / Image transformation information
        generator = {}
        generator_params = ['rotation_range', 'width_shift_range',
                            'height_shift_range', 'shear_range',
                            'zoom_range', 'horizontal_flip',
                            'vertical_flip']
        default_params = [30, 0.1, 0.1, 0, 0.1, True, False]
        for param, default_value in zip(generator_params, default_params):
            if param in doc:
                generator[param] = doc[param]
            else:
                generator[param] = default_value

        generator['generator_callable'] = standard_generators

        params = {
            'data': types.DataConfig(**data_info),
            'model': types.ModelConfig(**model),
            'generator': types.GeneratorConfig(**generator),
            'batch_size': doc.batch_size,
            'seed': 999,
            'val_split': 0.1,
            'max_epochs': 100,
            'early_stopping': True,
            'reduce_lr': False,
            'job_fn': None,
            'job_name': None
        }

        data['params'] = types.ParamConfig(**params)
        docs_to_train.append(data)
    return docs_to_train


def __get_data_if_not_exists(gcs_dir, local_dir):
    if not os.path.isdir(local_dir):
        logging.info(('Dataset {} does not exist. '
                      'Downloading from GCS...'.format(local_dir)))
        os.mkdir(local_dir)
        exit = os.system(
            'gsutil -m rsync -r -d {} {}'.format(gcs_dir, local_dir))
        return exit == 0
    return True


def __load_data(params):
    return preprocessing.prepare_data(params)


def __train_model(params, x_train, y_train, x_valid, y_valid,
                  num_gpu=0):
    train_gen, valid_gen = params.generator.generator_callable(
        x_train, y_train,
        x_valid, y_valid,
        params.batch_size,
        **params.generator.__dict__
    )

    model = params.model.model_callable(input_shape=x_train.shape[1:],
                                        num_classes=y_train.shape[1],
                                        **params.model.__dict__)
    model_original = model
    if num_gpu > 0:
        model = multi_gpu_model(model, gpus=num_gpu)
    metrics = ['acc',
               utils.sensitivity,
               utils.specificity,
               utils.true_positives,
               utils.false_negatives]
    model.compile(optimizer=params.model.optimizer,
                  loss=params.model.loss,
                  metrics=metrics)
    callbacks = utils.create_callbacks(x_train, y_train, x_valid, y_valid,
                                       early_stopping=params.early_stopping,
                                       reduce_lr=params.reduce_lr)
    history = model.fit_generator(train_gen,
                                  epochs=params.max_epochs,
                                  validation_data=valid_gen,
                                  verbose=2,
                                  callbacks=callbacks)
    return model_original, history


def evaluate_model(x_test, y_test, model,
                   normalize=True, x_train=None,
                   num_gpus=0):
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

    if num_gpus > 0:
        model = multi_gpu_model(model)
    results = model.evaluate(x=x_test, y=y_test, batch_size=1,
                             verbose=1)
    return results


def parse_args(args):
    parser = argparse.ArgumentParser(description='Evaluation script.')
    subparsers = parser.add_subparsers(
        help='Arguments for specific evaluation types.',
        dest='eval_type'
    )
    subparsers.required = True

    param_parser = subparsers.add_parser('param-list')
    param_parser.add_argument(
        'param-list-config',
        help=('Path to config file. This file must have a list of '
              'ParamConfig objects stored in EVAL_PARAM_LIST.')
    )

    kibana_parser = subparsers.add_parser('kibana')
    kibana_parser.add_argument('--address',
                               help='Address to access Kibana.',
                               default='http://104.196.51.205')
    kibana_parser.add_argument('--lower',
                               help='Lower bound of best_val_acc to search.',
                               default='0.85')
    kibana_parser.add_argument('--upper',
                               help='Upper bound of best_val_acc to search.',
                               default='0.93')

    parser.add_argument(
        '--gpu',
        help=('Ids of the GPU to use (as reported by nvidia-smi). '
              'Separated by comma, no spaces. e.g. 0,1'),
        default=None
    )
    parser.add_argument(
        '--log-dir',
        help=('Location to store logs.'),
        default='../logs/'
    )
    parser.add_argument(
        '--config',
        help=('Configuration file, if you want to specify GPU and '
              'log directories there.'),
        default=None
    )
    parser.add_argument(
        '--num-iterations',
        help=('Number of times to test a parameter config. '
              'Evaluation results are averaged.'),
        default=1
    )
    parser.add_argument(
        '--slack-token',
        help=('Slack token, to upload results to slack.'),
        default=None
    )

    return parser.parse_args(args)


def check_config(config):
    logging.info('Checking that config has all required attributes')
    logging.debug('EVAL_PARAM_LIST: {}'.format(config.EVAL_PARAM_LIST))
    if (not (isinstance(config.EVAL_PARAM_LIST, list)) or
       not (isinstance(config.EVAL_PARAM_LIST[0], types.ParamConfig))):
        raise ValueError('EVAL_PARAM_LIST must be a list of ParamConfig')


def iterate_eval(num_iterations, params, num_gpu,
                 job_name=None, job_date=None,
                 purported_loss=None, purported_accuracy=None,
                 purported_sensitivity=None,
                 slack_token=None):
    """
    Beautiful piece of text that has more logging than code.
    """
    result_list = []
    for i in range(num_iterations):
        logging.info("-----Iteration {}-----".format(i + 1))
        params.seed = random.randint(0, 1000000)
        logging.info("Using seed {}".format(params.seed))
        (x_train, x_valid, x_test, y_train, y_valid, y_test,
         _, _, _) = __load_data(params)
        model, history = __train_model(params, x_train, y_train,
                                       x_valid, y_valid,
                                       num_gpu=num_gpu)
        result = evaluate_model(x_test, y_test, model,
                                normalize=True, x_train=x_train)
        result_list.append(result)
        logging.info("-----Results-----")
        logging.info('Loss: {}'.format(result[0]))
        logging.info('Acc: {}'.format(result[1]))
        logging.info('Sensitivity: {}'.format(result[2]))
        logging.info('Specificity: {}'.format(result[3]))
        logging.info('True Positives: {}'.format(result[4]))
        logging.info('False Negatives: {}'.format(result[5]))

        if (slack_token is not None):
            text = "-----Iteration {}-----\n".format(i + 1)
            text += "Seed: {}\n".format(params.seed)
            text += "Params: {}\n".format(params)
            if (job_name is not None):
                text += 'Job name: {}\n'.format(job_name)
                text += 'Job date: {}\n'.format(job_date)
                text += 'Purported accuracy: {}\n'.format(
                    purported_accuracy)
                text += 'Purported loss: {}\n'.format(
                    purported_loss)
                text += 'Purported sensitivity: {}\n'.format(
                    purported_sensitivity)
            text += "\n-----Results-----\n"
            text += 'Loss: {}\n'.format(result[0])
            text += 'Acc: {}\n'.format(result[1])
            text += 'Sensitivity: {}\n'.format(result[2])
            text += 'Specificity: {}\n'.format(result[3])
            text += 'True Positives: {}\n'.format(result[4])
            text += 'False Negatives: {}\n'.format(result[5])
            slack.write_to_slack(text, slack_token)

    result_list = np.average(result_list, axis=0)
    logging.info("---------------Final Results---------------")
    logging.info('Loss: {}'.format(result_list[0]))
    logging.info('Acc: {}'.format(result_list[1]))
    logging.info('Sensitivity: {}'.format(result_list[2]))
    logging.info('Specificity: {}'.format(result_list[3]))
    logging.info('True Positives: {}'.format(result_list[4]))
    logging.info('False Negatives: {}'.format(result_list[5]))

    if (slack_token is not None):
            text = "-----Final Results-----\n"
            text += "Seed: {}\n".format(params.seed)
            text += "Params: {}\n".format(params)
            if (job_name is not None):
                text += 'Job name: {}\n'.format(job_name)
                text += 'Job date: {}\n'.format(job_date)
                text += 'Purported accuracy: {}\n'.format(
                    purported_accuracy)
                text += 'Purported loss: {}\n'.format(
                    purported_loss)
                text += 'Purported sensitivity: {}\n'.format(
                    purported_sensitivity)
            text += "\n-----Average Results-----\n"
            text += 'Loss: {}\n'.format(result_list[0])
            text += 'Acc: {}\n'.format(result_list[1])
            text += 'Sensitivity: {}\n'.format(result_list[2])
            text += 'Specificity: {}'.format(result_list[3])
            text += 'True Positives: {}\n'.format(result_list[4])
            text += 'False Negatives: {}\n'.format(result_list[5])
            slack.write_to_slack(text, slack_token)


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Set config if exists
    if args.config:
        user_config = importlib.import_module(args.config)

    # Set logger
    if args.config is not None and user_config.LOG_DIR is not None:
        parent_log_file = pathlib.Path(
            user_config.LOG_DIR) / 'eval-results-{}.txt'.format(
            datetime.datetime.utcnow().isoformat()
        )
    else:
        parent_log_file = pathlib.Path(
            args.log_dir) / 'eval-results-{}.txt'.format(
            datetime.datetime.utcnow().isoformat()
        )
    logger.configure_parent_logger(parent_log_file, level=logging.INFO)

    # Choose GPU to use
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        num_gpu = len(args.gpu.split(','))
    elif args.config is not None and user_config.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = user_config.gpus
        num_gpu = len(user_config.gpus.split(','))
    else:
        num_gpu = 0

    # Number of iterations
    if args.config is not None and user_config.num_iterations is not None:
        num_iterations = user_config.num_iterations
    else:
        num_iterations = args.num_iterations

    # Slack token
    if args.config is not None and user_config.SLACK_TOKEN is not None:
        slack_token = user_config.SLACK_TOKEN
    else:
        slack_token = args.slack_token

    if args.eval_type == 'kibana':
        # Fetch params list from Kibana to evaluate
        models = get_models_to_train(args.address, args.lower,
                                     args.upper, '../tmp')
        for model in models:
            logging.info("----------------Evaluation-------------------")
            logging.info('Job name: {}'.format(model['job_name']))
            logging.info('Job date: {}'.format(model['date']))
            logging.info('Purported accuracy: {}'.format(
                model['purported_accuracy']))
            logging.info('Purported loss: {}'.format(
                model['purported_loss']))
            logging.info('Purported sensitivity: {}'.format(
                model['purported_sensitivity']))
            params = model['params']
            __get_data_if_not_exists(params.data.gcs_url,
                                     model['local_dir'])
            iterate_eval(num_iterations, params, num_gpu,
                         job_name=model['job_name'],
                         job_date=model['date'],
                         purported_accuracy=model['purported_accuracy'],
                         purported_loss=model['purported_loss'],
                         purported_sensitivity=model['purported_sensitivity'],
                         slack_token=slack_token)

    else:
        # Manual evaluation of a list of ParamConfig
        logging.info('Using config {}'.format(args.param_list_config))
        param_list_config = importlib.import_module(args.param_list_config)
        check_config(param_list_config)

        params = param_list_config.EVAL_PARAM_LIST
        for param in params:
            __get_data_if_not_exists(param.data.gcs_url,
                                     param.data.data_dir)
            iterate_eval(num_iterations, param, num_gpu,
                         slack_token=slack_token)


if __name__ == '__main__':
    main()
