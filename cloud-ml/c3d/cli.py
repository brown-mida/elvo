import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    args = parser.parse_args()

    if args.command == 'upload':
        # Upload the files in this directory to GCS
        subprocess.call(['python3', 'setup.py', 'sdist'])
        subprocess.call(
            ['gsutil', 'rsync', '-r', '.', 'gs://elvos/cloud-ml/cloud3d'])
    if args.command == 'train-local':
        subprocess.call([
            'gcloud', 'ml-engine', 'local', 'train',
            '--module-name=trainer.task',
            '--package-path==trainer',
        ])
