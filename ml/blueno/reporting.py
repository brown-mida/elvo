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

    epochs = elasticsearch_dsl.Integer()
    train_acc = elasticsearch_dsl.Float()
    final_val_acc = elasticsearch_dsl.Float()
    best_val_acc = elasticsearch_dsl.Float()
    final_val_loss = elasticsearch_dsl.Float()
    best_val_loss = elasticsearch_dsl.Float()
    final_val_sensitivity = elasticsearch_dsl.Float()
    best_val_sensitivity = elasticsearch_dsl.Float()
    final_val_auc = elasticsearch_dsl.Float()

    class Index:
        name = TRAINING_JOBS


def insert_or_ignore_filepaths(log_file: pathlib.Path,
                               csv_file: typing.Optional[pathlib.Path],
                               gpu1708=False):
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

    if author is None and gpu1708:
        author = _fill_author_gpu1708(created_at, job_name)
    ended_at = _extract_ended_at(log_file)
    model_url = _extract_model_url(log_file)
    final_val_auc = _extract_auc(log_file)

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
                                     final_val_auc=final_val_auc)
        insert_or_ignore(training_job)
    except (ValueError, EmptyDataError):
        print('metrics file {} is empty'.format(csv_file))
        return


def insert_or_ignore(training_job):
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
        training_job.save()
    else:
        print('job {} created at {} exists'.format(
            training_job.job_name, training_job.created_at))


def construct_job(job_name,
                  created_at,
                  params,
                  raw_log,
                  metrics,
                  metrics_filename,
                  author=None,
                  ended_at=None,
                  model_url=None,
                  final_val_auc=None) -> TrainingJob:
    """
    Constructs a training job object from the given parameters.

    Note that these parameters are experimental.
    :param job_name:
    :param created_at:
    :param params:
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
                               final_val_auc=final_val_auc)

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
        first_line = f.readline()
        if first_line.endswith('using params:\n'):
            second_line = f.readline()
            return second_line
        return None


def _extract_author(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        f.readline()
        f.readline()
        third_line = f.readline()
        if 'INFO - author:' in third_line:
            author = third_line.rstrip('\n').split(':')[-1].strip()
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
