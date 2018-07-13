import keras
import os
import pytest
from sklearn import model_selection

import generators.luke
import models.luke
from blueno.types import ParamGrid


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_from_dict():
    param_dict = {
        'data': [
            {
                'data_dir': f'/gpfs/main/home/lzhu7/elvo-analysis/data'
                            'processed-standard/arrays/',
                'labels_path': f'/gpfs/main/home/lzhu7/elvo-analysis/data'
                               'processed-standard/labels.csv',
                'index_col': 'Anon ID',
                'label_col': 'occlusion_exists',
            },
        ],

        'seed': [42, 0],
        'val_split': [0.2, 0.1],

        'generator': [{
            'generator_callable': generators.luke.standard_generators,
            'rotation_range': 30,
        }],

        'batch_size': [8],

        'model': model_selection.ParameterGrid({
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
        }),
    }
    assert ParamGrid(**param_dict) is not None
