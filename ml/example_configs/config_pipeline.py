import pathlib

import keras
from sklearn import model_selection

import blueno
import generators.luke
import models.luke
import preprocessors.luke

USER = ''  # ex: luke
BLUENO_HOME = ''  # ex: pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis')

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
    'pipeline_callable': [preprocessors.luke.preprocess_data],
    'data_dir': [str(pathlib.Path(DATA_DIR) / 'numpy_compressed')],
    'labels_path': [str(pathlib.Path(DATA_DIR) / 'metadata')],
    'index_col': ['Anon ID'],
    'label_col': ['occlusion_exists'],
    'gcs_url': [None],
    'height_offset': [30, 40, 50],
    'mip_thickness': [24],
    'pixel_value_range': [(0, 400), (0, 300), (0, 200)],
}))

data_list = [blueno.LukePipelineConfig(**d) for d in data_list]

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

PARAM_GRID = [blueno.ParamConfig(**p) for p in PARAM_GRID]
