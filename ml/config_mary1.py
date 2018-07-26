import pathlib

import keras
from sklearn import model_selection

import blueno
import generators.luke
import models.luke

USER = 'mary'
BLUENO_HOME = '/home/lzhu7/elvo-analysis/'
DATA_DIR = f'{BLUENO_HOME}data/'
LOG_DIR = f'{BLUENO_HOME}logs/'

NUM_GPUS = 1
GPU_OFFSET = 2

# SLACK_TOKEN = 'xoxp-314216549302-331430419907-396979178437-' \
#               'ae769a026a3c0f91623e9a6565f0d9ee'

model_list = list(model_selection.ParameterGrid({
    'model_callable': [models.luke.resnet],
    'dropout_rate1': [0.7],
    'dropout_rate2': [0.7],
    'optimizer': [
        keras.optimizers.Adam(lr=1e-5),
    ],
    'loss': [
        keras.losses.categorical_crossentropy,
        keras.losses.binary_crossentropy,
    ],
    'freeze': [False],
}))

model_list = [blueno.ModelConfig(**m) for m in model_list]

PARAM_GRID = model_selection.ParameterGrid({
    'data': [blueno.DataConfig(
        data_dir=str(pathlib.Path(DATA_DIR) /
                     'processed-lower/arrays/'),
        labels_path=str(
            pathlib.Path(DATA_DIR) / 'processed-lower/labels.csv'),
        index_col='Anon ID',
        label_col='occlusion_exists',
        gcs_url='gs://elvos/processed/processed-lower')
    ],
    'generator': [blueno.GeneratorConfig(
        generator_callable=generators.luke.standard_generators,
        rotation_range=30)],
    'model': model_list,
    'batch_size': [8],
    'seed': [42, 0],
    'val_split': [0.2, 0.1],
})

PARAM_GRID = [blueno.ParamConfig(**p) for p in PARAM_GRID]
