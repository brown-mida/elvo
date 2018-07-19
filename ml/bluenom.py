"""
Script to load the data in the logs directory of gpu1708 up to
Elasticsearch.
"""
import argparse
import pathlib

import os
from elasticsearch_dsl import connections

from blueno.elasticsearch import (
    insert_or_ignore_filepaths, JOB_INDEX,
    TrainingJob,
)


def bluenom(log_dir: pathlib.Path, gpu1708=False):
    """
    Uploads logs in the directory to bluenom. This will
    only upload logs which have uploaded to Slack.

    This is idempotent so it can be run multiple times.
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
            insert_or_ignore_filepaths(file_path,
                                       metrics_file_path,
                                       gpu1708)
        else:
            print('{} is not a log or CSV file'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', default=False, type=bool)
    parser.add_argument('--gpu1708', default=False, type=bool)
    args = parser.parse_args()

    # Creates a connection to our Airflow instance
    connections.create_connection(hosts=['http://104.196.51.205'])

    if args.reset:
        print('resetting index')
        if JOB_INDEX.exists():
            JOB_INDEX.delete()
        JOB_INDEX.create()

    TrainingJob.init()
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs')
    bluenom(path, gpu1708=args.gpu1708)
