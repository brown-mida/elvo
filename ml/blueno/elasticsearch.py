import ast
import pathlib
import typing
from collections.__init__ import namedtuple

import elasticsearch_dsl
import pandas as pd
import re
from pandas.errors import EmptyDataError

TRAINING_JOBS = 'training_jobs'
JOB_INDEX = elasticsearch_dsl.Index(TRAINING_JOBS)

Metrics = namedtuple('Metrics', ['epochs',
                                 'train_acc',
                                 'final_val_acc',
                                 'best_val_acc',
                                 'final_val_loss',
                                 'best_val_loss',
                                 'final_val_sensitivity',
                                 'best_val_sensitivity'])


class TrainingJob(elasticsearch_dsl.Document):
    id = elasticsearch_dsl.Integer()
    schema_version = elasticsearch_dsl.Integer()
    job_name = elasticsearch_dsl.Keyword()
    author = elasticsearch_dsl.Keyword()
    created_at = elasticsearch_dsl.Date()
    ended_at = elasticsearch_dsl.Date()
    params = elasticsearch_dsl.Text()
    raw_log = elasticsearch_dsl.Text()
    model_url = elasticsearch_dsl.Text()

    # Metrics
    epochs = elasticsearch_dsl.Integer()
    train_acc = elasticsearch_dsl.Float()
    final_val_acc = elasticsearch_dsl.Float()
    best_val_acc = elasticsearch_dsl.Float()
    final_val_loss = elasticsearch_dsl.Float()
    best_val_loss = elasticsearch_dsl.Float()
    final_val_sensitivity = elasticsearch_dsl.Float()
    best_val_sensitivity = elasticsearch_dsl.Float()
    final_val_auc = elasticsearch_dsl.Float()
    best_val_auc = elasticsearch_dsl.Float()

    # Params
    batch_size = elasticsearch_dsl.Integer()
    val_split = elasticsearch_dsl.Float()
    seed = elasticsearch_dsl.Integer()

    rotation_range = elasticsearch_dsl.Float()
    width_shift_range = elasticsearch_dsl.Float()
    height_shift_range: float = elasticsearch_dsl.Float()
    shear_range = elasticsearch_dsl.Float()
    zoom_range = elasticsearch_dsl.Float()
    horizontal_flip = elasticsearch_dsl.Boolean()
    vertical_flip = elasticsearch_dsl.Boolean()

    dropout_rate1 = elasticsearch_dsl.Float()
    dropout_rate2 = elasticsearch_dsl.Float()

    data_dir = elasticsearch_dsl.Keyword()
    gcs_url = elasticsearch_dsl.Keyword()

    # We need to keep a list of params for the parser because
    # we can't use traditional approaches to get the class attrs
    params_to_parse = ['batch_size',
                       'val_split',
                       'seed',
                       'rotation_range',
                       'width_shift_range',
                       'height_shift_range',
                       'shear_range',
                       'zoom_range',
                       'horizontal_flip',
                       'vertical_flip',
                       'dropout_rate1',
                       'dropout_rate2',
                       'data_dir',
                       'gcs_url',
                       'mip_thickness',
                       'height_offset',
                       'pixel_value_range']

    class Index:
        name = TRAINING_JOBS


def insert_or_ignore_filepaths(log_file: pathlib.Path,
                               csv_file: typing.Optional[pathlib.Path],
                               gpu1708=False,
                               clean_job_names=False,
                               alias='default'):
    """
    Parses matching log file and csv and uploads the file up to the
    Elasticsearch index, if it doesn't exist.

    Note that the parsing is very brittle, so important logs
    should be documented in bluenot.py

    :param log_file:
    :param csv_file:
    :param gpu1708:
    :return:
    """
    filename = str(log_file.name)

    job_name, created_at = _parse_filename(filename)
    params = _extract_params(log_file)
    author = _extract_author(log_file)
    raw_log = open(log_file).read()
    ended_at = _extract_ended_at(log_file)
    model_url = _extract_model_url(log_file)
    final_val_auc = _extract_auc(log_file)
    best_val_auc = _extract_best_auc(log_file)

    if author is None and gpu1708:
        author = _fill_author_gpu1708(created_at, job_name)

    if params:
        params_dict = _parse_params_str(params)
    else:
        params_dict = None

    if clean_job_names and params_dict:
        num_classes = job_name.split('_')[-1]
        data_dir = params_dict['data_dir']
        # Ignore this dir at this means preprocessing is being done
        if 'numpy_compressed' not in data_dir:
            as_path = pathlib.Path(data_dir)
            job_name = f'{as_path.parent.name}_{num_classes}'

    try:
        metrics = _extract_metrics(csv_file)
        training_job = construct_job(job_name,
                                     created_at,
                                     params,
                                     raw_log,
                                     metrics,
                                     str(csv_file.name),
                                     author=author,
                                     ended_at=ended_at,
                                     model_url=model_url,
                                     final_val_auc=final_val_auc,
                                     best_val_auc=best_val_auc,
                                     params_dict=params_dict)
        insert_or_ignore(training_job, alias=alias)
    except (ValueError, EmptyDataError):
        print('metrics file {} is empty'.format(csv_file))
        return


def insert_or_ignore(training_job: TrainingJob, alias='default'):
    """Inserts the training job into the elasticsearch index
    if no job with the same name and creation timestamp exists.
    """
    matches = JOB_INDEX.search() \
        .query('match', job_name=training_job.job_name) \
        .query('match', created_at=training_job.created_at) \
        .count()

    if 'slack' not in training_job.raw_log:
        print('job is incomplete, returning')
        return

    if matches == 0:
        training_job.save(using=alias)
    else:
        print('job {} created at {} exists'.format(
            training_job.job_name, training_job.created_at))


def construct_job(job_name,
                  created_at,
                  params: str,
                  raw_log,
                  metrics,
                  metrics_filename,
                  author=None,
                  ended_at=None,
                  model_url=None,
                  final_val_auc=None,
                  best_val_auc=None,
                  params_dict=None) -> TrainingJob:
    """
    Constructs a training job object from the given parameters.

    Note that these parameters are experimental.
    :param job_name:
    :param created_at:
    :param params: a string of bluenot config params
    :param raw_log:
    :param metrics:
    :param metrics_filename:
    :param author:
    :param ended_at:
    :param model_url:
    :param final_val_auc:
    :return:
    """
    training_job = TrainingJob(schema_version=1,
                               job_name=job_name,
                               author=author,
                               created_at=created_at,
                               ended_at=ended_at,
                               params=params,
                               raw_log=raw_log,
                               model_url=model_url,
                               final_val_auc=final_val_auc,
                               best_val_auc=best_val_auc)

    if params_dict:
        for key, val in params_dict.items():
            if key is 'data_dir' and val.endswith('/'):  # standardize dirpaths
                val = val[:-1]
            training_job.__setattr__(key, val)

    if (job_name, created_at) == _parse_filename(metrics_filename):
        print('found matching CSV file, setting metrics')
        training_job.epochs = metrics.epochs
        training_job.train_acc = metrics.train_acc
        training_job.final_val_acc = metrics.final_val_acc
        training_job.best_val_acc = metrics.best_val_acc
        training_job.final_val_loss = metrics.final_val_loss
        training_job.best_val_loss = metrics.best_val_loss
        training_job.final_val_sensitivity = \
            metrics.final_val_sensitivity
        training_job.best_val_sensitivity = \
            metrics.best_val_sensitivity
    return training_job


def _parse_filename(filename: str) -> typing.Tuple[str, str]:
    """Parse a CSV or log file.
    """
    date_start_idx = filename.find('2018')
    job_name = filename[:date_start_idx - 1]
    created_at = filename[date_start_idx:-4]
    return job_name, created_at


def _extract_params(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if line.endswith('using params:\n'):
                second_line = f.readline()
                return second_line
        return None


def _extract_author(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if 'INFO - author:' in line:
                author = line.rstrip('\n').split(':')[-1].strip()
                return author
    return None


def _extract_ended_at(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if 'INFO - end time:' in line:
                return line.split(' ')[-1].rstrip('\n')
    return None


def _extract_model_url(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if 'uploading model' in line:
                return line.split(' ')[-1].rstrip('\n')
    return None


def _extract_auc(log_path: pathlib.Path) -> typing.Optional[float]:
    with open(log_path) as f:
        pattern = re.compile(r'initial_comment=Accuracy%3A\+(.+)'
                             r'AUC%3A\+(.+?)%0')
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(2))
    return None


def _extract_best_auc(log_path: pathlib.Path) -> typing.Optional[float]:
    with open(log_path) as f:
        for line in f:
            if 'INFO - val_auc:' in line:
                return float(line.split(' ')[-1].rstrip('\n'))


def _parse_params_str(params_str: str) -> typing.Dict[str, typing.Any]:
    """Parses the param string outputs that most logs contain.

    This code is very rigid, and will likely break.
    """
    param_dict = {}
    for param in TrainingJob.params_to_parse:
        if param in ('zoom_range', 'pixel_value_range'):
            pattern = r'{}=([^,]+, [^,]+)[,)]'.format(param)
            match = re.search(pattern, params_str)
            if match:
                param_dict[param] = match.group(1)
        else:
            patterns = [r'{}=(.*?)[,)]'.format(param),
                        r"'{}'".format(param) + r": (.*?)[,}]"]
            for pattern in patterns:
                match = re.search(pattern, params_str)
                if match:
                    value_str = match.group(1)
                    value = ast.literal_eval(value_str)
                    param_dict[param] = value
    print('parsed params:', param_dict)
    return param_dict


def _extract_metrics(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    return Metrics(epochs=df['epoch'].max(),
                   train_acc=df['acc'].iloc[-1],
                   final_val_acc=df['val_acc'].iloc[-1],
                   best_val_acc=df['val_acc'].max(),
                   final_val_loss=df['val_loss'].iloc[-1],
                   best_val_loss=df['val_loss'].min(),
                   final_val_sensitivity=df['val_sensitivity'].iloc[-1],
                   best_val_sensitivity=df['val_sensitivity'].max())


def _fill_author_gpu1708(created_at, job_name):
    if created_at > '2018-07-11' \
            or 'processed_' in job_name \
            or 'processed-no-vert' in job_name:
        author = 'sumera'
    else:
        author = 'luke'
    return author
