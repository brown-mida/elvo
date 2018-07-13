import pathlib

import os
from elasticsearch_dsl import connections

from blueno.reporting import insert_job_by_filepaths, JOB_INDEX, TrainingJob

connections.create_connection(hosts=['http://104.196.51.205'])


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


if __name__ == '__main__':
    print('resetting job index')
    if JOB_INDEX.exists():
        JOB_INDEX.delete()
    JOB_INDEX.create()
    TrainingJob.init()
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs')
    bluenom(path, gpu1708=True)
