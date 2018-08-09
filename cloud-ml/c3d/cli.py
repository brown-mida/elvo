import argparse
import subprocess

import warnings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--job', required=False)
    args = parser.parse_args()

    warnings.warn('Make sure the working directory is c3d.')

    if args.command == 'upload-c3d':
        # Upload the files in this directory to GCS
        subprocess.call(['python3', 'setup.py', 'sdist'])
        subprocess.call(
            ['gsutil', 'rsync', '-r', 'dist/', 'gs://elvos/cloud-ml/c3d'])
    elif args.command == 'upload-blueno':
        # Upload the files in this directory to GCS
        subprocess.call(['python3', 'setup.py', 'sdist'], cwd='../../')
        subprocess.call(
            ['gsutil', 'rsync', '-r', '../../dist/', 'gs://elvos/cloud-ml/'])
    elif args.command == 'train-local':
        subprocess.call([
            'gcloud', 'ml-engine', 'local', 'train',
            '--module-name=trainer.task',
            '--package-path==trainer',
        ])
    elif args.command == 'train-cloud':
        if not args.job:
            raise ValueError('--job flag not specified')
        warnings.warn('Make sure you the packages on the cloud have'
                      ' been uploaded recently')
        packages = ['gs://elvos/cloud-ml/blueno-0.1.0.tar.gz',
                    'gs://elvos/cloud-ml/c3d/cloudml-c3d-0.0.2.tar.gz']
        subprocess.call([
            'gcloud', 'ml-engine', 'jobs', 'submit', 'training',
            args.job,
            '--module-name=trainer.task',
            '--packages={}'.format(','.join(packages)),
            '--python-version=3.5',
            '--region=us-east1',
            '--runtime-version=1.4',
            '--stream-logs',
            '--scale-tier=BASIC_GPU',
        ])
