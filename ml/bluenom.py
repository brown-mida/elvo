import pathlib
import typing
from collections import namedtuple

import elasticsearch_dsl
import os
import pandas as pd
from elasticsearch_dsl import connections
from pandas.errors import EmptyDataError

TRAINING_JOBS = 'training_jobs'
JOB_INDEX = elasticsearch_dsl.Index(TRAINING_JOBS)

connections.create_connection(hosts=['http://104.196.51.205'])


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

    class Index:
        name = TRAINING_JOBS


Metrics = namedtuple('Metrics', ['epochs',
                                 'train_acc',
                                 'final_val_acc',
                                 'best_val_acc',
                                 'final_val_loss',
                                 'best_val_loss',
                                 'final_val_sensitivity',
                                 'best_val_sensitivity'])


def bluenom(log_dir: pathlib.Path, gpu1708=False):
    """
    Uploads logs in the directory to bluenom.

    This is not idempotent so the index should be deleted before this is
    run again.
    :return:
    """

    metrics_file_path: pathlib.Path = None

    # noinspection PyTypeChecker
    for filename in sorted(os.listdir(log_dir)):
        file_path = log_dir / filename

        if filename.endswith('.csv'):
            metrics_file_path = log_dir / filename
        elif filename.endswith('.log'):
            print('indexing {}'.format(filename))
            insert_job_by_filepaths(file_path,
                                    metrics_file_path,
                                    gpu1708)
        else:
            print('{} is not a log or CSV file'.format(filename))


def insert_job_by_filepaths(log_file: pathlib.Path,
                            csv_file: typing.Optional[pathlib.Path],
                            gpu1708=False):
    filename = str(log_file.name)

    job_name, created_at = _parse_filename(filename)
    params = _extract_params(log_file)
    author = _extract_author(log_file)
    raw_log = open(log_file).read()

    if author is None and gpu1708:
        author = _fill_author_gpu1708(created_at, job_name)
    ended_at = _extract_ended_at(log_file)
    model_url = _extract_model_url(log_file)

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
                                     model_url=model_url)
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
                  model_url=None) -> TrainingJob:
    """Note that these parameters are experimental.
    """
    training_job = TrainingJob(schema_version=1,
                               job_name=job_name,
                               author=author,
                               created_at=created_at,
                               ended_at=ended_at,
                               params=params,
                               raw_log=raw_log,
                               model_url=model_url)

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


def _extract_params(path: pathlib.Path) -> typing.Optional[str]:
    with open(path) as f:
        first_line = f.readline()
        if first_line.endswith('using params:\n'):
            second_line = f.readline()
            return second_line
        return None


def _extract_author(path: pathlib.Path) -> typing.Optional[str]:
    with open(path) as f:
        f.readline()
        f.readline()
        third_line = f.readline()
        if 'INFO - author:' in third_line:
            author = third_line.rstrip('\n').split(':')[-1].strip()
            return author
    return None


def _extract_ended_at(path: pathlib.Path):
    with open(path) as f:
        lines = [line for line in f]
        if len(lines) > 0 and 'end time:' in lines[-1]:
            return lines[-1].split(' ')[-1].strip()
    return None


def _extract_model_url(path: pathlib.Path):
    with open(path) as f:
        lines = [line for line in f]
        for line in lines:
            if 'uploading model' in line:
                return line.split(' ')[-1]
    return None


def _extract_metrics(path: pathlib.Path):
    df = pd.read_csv(path)
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


if __name__ == '__main__':
    print('resetting job index')
    if JOB_INDEX.exists():
        JOB_INDEX.delete()
    JOB_INDEX.create()
    TrainingJob.init()
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs')
    bluenom(path, gpu1708=True)
