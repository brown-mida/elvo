import pathlib

USER = 'charlene'
BLUENO_HOME = '/research/rih-cs/datasets/elvo/v1/'
TRAIN_DIR = f'{BLUENO_HOME}train/'
TEST_DIR = f'{BLUENO_HOME}test/'
VAL_DIR = f'{BLUENO_HOME}validation/'

DATA_DIR = f'{BLUENO_HOME}preprocessed/'
LOG_DIR = f'{BLUENO_HOME}logs/'

# Split 1204 .npy files in data/ into train/test/validation
import argparse
import os
from shutil import copyfile

# the number of files in each subdirectory
TRAIN_DATA = 842
TEST_DATA = 1082
VAL_DATA = 1204

def split_data(abs_dirname):
    """Move files into subdirectories."""
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0

    for f in files:
        base_f = os.path.basename(f)
        # copy files to their respective directory
        if i <= TRAIN_DATA:
            copyfile(f, TRAIN_DIR + base_f)
        elif i <= TEST_DATA:
            copyfile(f, TEST_DIR + base_f)
        elif i <= VAL_DATA:
            copyfile(f, VAL_DIR + base_f)
        i += 1

def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Split data into train/test/val.')

    parser.add_argument('src_dir', help='source directory')

    return parser.parse_args()

def main():
    """Module's main entry point (zopectl.command)."""
    args = parse_args()
    src_dir = args.src_dir

    if not os.path.exists(src_dir):
        raise Exception('Directory does not exist ({0}).'.format(src_dir))

    split_data(os.path.abspath(src_dir))

if __name__ == '__main__':
    main()
