import pathlib

import keras
from sklearn import model_selection

import blueno
import generators.luke
import models.luke

USER = 'mary'
BLUENO_HOME = '/research/rih-cs/datasets/elvo-multiphase/v1/'
DATA_DIR = f'{BLUENO_HOME}preprocessed/phase2/'
LOG_DIR = f'{BLUENO_HOME}logs/'
SLACK_TOKEN = 'xoxp-314216549302-331430419907-396979178437-' \
              'ae769a026a3c0f91623e9a6565f0d9ee'

NUM_GPUS = 1
GPU_OFFSET = 2

# a lot of .npy files in data/ and preprocessed/ and labels.csv (patient_id, label 0 or 1)

model_list = list(model_selection.ParameterGrid({
    'model_callable': [models.luke.resnet],
    'dropout_rate1': [0.8],
    'dropout_rate2': [0.8],
    'optimizer': [
        keras.optimizers.Adam(lr=1e-5),
    ],
    'loss': [
        keras.losses.categorical_crossentropy,
    ],
    'freeze': [False],
}))

model_list = [blueno.ModelConfig(**m) for m in model_list]

PARAM_GRID = model_selection.ParameterGrid({
    'data': [blueno.DataConfig(
        data_dir=str(pathlib.Path(DATA_DIR) /
                     'processed-new-training-2/arrays/'),
        labels_path=str(
            pathlib.Path(DATA_DIR) / 'processed-new-training-2/labels.csv'),
        index_col='Anon ID',
        label_col='occlusion_exists',
        gcs_url='gs://elvos/processed/processed-new-training-2')
    ],
    'generator': [blueno.GeneratorConfig(
        generator_callable=generators.luke.standard_generators,
        rotation_range=30)],
    'model': model_list,
    'batch_size': [5],
    'seed': [0],
    'val_split': [0.1],  # So we run the grid 16 times
})

PARAM_GRID = [blueno.ParamConfig(**p) for p in PARAM_GRID]
