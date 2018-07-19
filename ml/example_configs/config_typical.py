import pathlib

import keras
from sklearn import model_selection

import blueno
import generators.luke
import models.luke

# Your name
USER = ''
BLUENO_HOME = pathlib.Path('< YOUR HOME DIRECTORY >')

DATA_DIR = ''
LOG_DIR = ''

NUM_GPUS = 1
GPU_OFFSET = 3

SLACK_TOKEN = ''

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

data_list = list(model_selection.ParameterGrid({
    'data_dir': [str(pathlib.Path(DATA_DIR) / 'processed-lower' / 'arrays')],
    'labels_path': [
        str(pathlib.Path(DATA_DIR) / 'processed-lower' / 'labels.csv')],
    'index_col': ['Anon ID'],
    'label_col': ['occlusion_exists'],
    'gcs_url': ['gs://elvos/processed/processed-lower'],

}))

data_list = [blueno.DataConfig(**d) for d in data_list]

PARAM_GRID = model_selection.ParameterGrid({
    'data': data_list,
    'generator': [blueno.GeneratorConfig(
        generator_callable=generators.luke.standard_generators,
        rotation_range=30)],
    'model': model_list,
    'batch_size': [8],
    'seed': [0, 1, 2, 3, 4, 5],
    'val_split': [0.1, 0.2, 0.3],
    'reduce_lr': [True, False],
    'early_stopping': [False],
    'max_epochs': [60],
})

# A list of ParamConfig objects to run jobs with
PARAM_GRID = [blueno.ParamConfig(**p) for p in PARAM_GRID]
