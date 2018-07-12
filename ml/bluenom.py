import pathlib
import typing
from collections import namedtuple

import elasticsearch_dsl
import os
import pandas as pd
from elasticsearch_dsl import connections
from pandas.errors import EmptyDataError

connections.create_connection(hosts=['http://104.196.51.205'])

TRAINING_JOBS = 'training_jobs'
JOB_INDEX = elasticsearch_dsl.Index(TRAINING_JOBS)


class TrainingJob(elasticsearch_dsl.Document):
    id = elasticsearch_dsl.Integer()
    schema_version = elasticsearch_dsl.Integer()
    job_name = elasticsearch_dsl.Keyword()
    author = elasticsearch_dsl.Keyword()
    created_at = elasticsearch_dsl.Date()
    params = elasticsearch_dsl.Text()
    raw_log = elasticsearch_dsl.Text()

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


def parse_filename(filename: str):
    """Parse a CSV or log file."""
    date_start_idx = filename.find('2018')
    job_name = filename[:date_start_idx - 1]
    created_at = filename[date_start_idx:-4]
    return job_name, created_at


def extract_params(path: pathlib.Path) -> typing.Optional[str]:
    with open(path) as f:
        first_line = f.readline()
        if first_line.endswith('using params:\n'):
            second_line = f.readline()
            return second_line
        return None


Metrics = namedtuple('Metrics', ['epochs',
                                 'train_acc',
                                 'final_val_acc',
                                 'best_val_acc',
                                 'final_val_loss',
                                 'best_val_loss',
                                 'final_val_sensitivity',
                                 'best_val_sensitivity'])


def extract_metrics(path: pathlib.Path):
    df = pd.read_csv(path)
    return Metrics(epochs=df['epoch'].max(),
                   train_acc=df['acc'].iloc[-1],
                   final_val_acc=df['val_acc'].iloc[-1],
                   best_val_acc=df['val_acc'].max(),
                   final_val_loss=df['val_loss'].iloc[-1],
                   best_val_loss=df['val_loss'].min(),
                   final_val_sensitivity=df['val_sensitivity'].iloc[-1],
                   best_val_sensitivity=df['val_sensitivity'].max())


def construct_job(job_name,
                  created_at,
                  params,
                  raw_log,
                  metrics,
                  metrics_filename,
                  author=None):
    if author is None:
        raise ValueError('Author must be specified.')
    training_job = TrainingJob(schema_version=1,
                               job_name=job_name,
                               author=author,
                               created_at=created_at,
                               params=params,
                               raw_log=raw_log)

    if (job_name, created_at) == parse_filename(metrics_filename):
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


def bluenom(log_dir: pathlib.Path):
    """
    Uploads logs in the directory to bluenom.

    This is not idempotent so the index should be deleted before this is
    run again.
    :return:
    """

    metrics: Metrics = None
    metrics_filename: str = ''

    unique_ids = set()
    for filename in sorted(os.listdir(log_dir)):
        file_path = log_dir / filename

        if filename.endswith('.csv'):
            try:
                metrics = extract_metrics(file_path)
                metrics_filename = filename
            except EmptyDataError:
                print('metrics file {} is empty'.format(metrics_filename))
        elif filename.endswith('.log'):
            print('indexing {}'.format(filename))
            job_name, created_at = parse_filename(filename)

            id_ = '{}|{}'.format(job_name, created_at)
            if id_ in unique_ids:
                print('{} already is in the index')
            else:
                unique_ids.add(id_)

                params = extract_params(file_path)
                raw_log = open(file_path).read()
                training_job = construct_job(job_name,
                                             created_at,
                                             params,
                                             raw_log,
                                             metrics,
                                             metrics_filename)

                training_job.save()
        else:
            print('{} is not a log or CSV file'.format(filename))


if __name__ == '__main__':
    if not JOB_INDEX.exists():
        print('creating index jobs')
        JOB_INDEX.create()
    TrainingJob.init()
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs')
    bluenom(path)
