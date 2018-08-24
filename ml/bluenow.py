"""
The script called by the web trainer.

Do not change this filename without updating train_dag as well
"""
import argparse
import logging
import pathlib

import keras
import os

import blueno
import bluenot
import generators.luke
import models.luke


# Code forked from bluenot.py, since that code is being changed by
# Andrew, we'll not do anything for now
def run_web_gpu1708_job(data_name: str,
                        batch_size: int,
                        val_split: float,
                        max_epochs: int,
                        job_name: str,
                        author_name: str,
                        three_fold_split: bool):
    blueno_home = pathlib.Path('/home/lzhu7/elvo-analysis')

    data_dir = blueno_home / 'data'
    log_dir = blueno_home / 'logs'

    param_config = blueno.ParamConfig(
        data=blueno.DataConfig(
            data_dir=str(data_dir / data_name / 'arrays'),
            labels_path=str(data_dir / data_name / 'labels.csv'),
            index_col='Anon ID',
            label_col='occlusion_exists',
            gcs_url=f'gs://elvos/processed/{data_name}',
        ),
        generator=blueno.GeneratorConfig(
            generator_callable=generators.luke.standard_generators,
        ),
        model=blueno.ModelConfig(
            model_callable=models.luke.resnet,
            optimizer=keras.optimizers.Adam(lr=1e-5),
            loss=keras.losses.categorical_crossentropy,
        ),
        batch_size=int(batch_size),
        seed=0,
        val_split=float(val_split),
        early_stopping=False,
        max_epochs=int(max_epochs),
        job_name=job_name,
        three_fold_split=three_fold_split,
    )

    logging.info('training web job {}'.format(param_config))

    # As noted in the rewrite, this function actually downloads the
    # new data to gpu1708
    # TODO(luke): Should the data be removed
    bluenot.check_data_in_sync(param_config)

    arrays = blueno.preprocessing.prepare_data(
        param_config, train_test_val=param_config.three_fold_split)
    if param_config.three_fold_split:
        (x_train, x_valid, x_test, y_train, y_valid, y_test, id_train,
         id_valid, id_test) = arrays
    else:
        x_train, x_valid, y_train, y_valid, id_train, id_valid = arrays

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    bluenot.start_job(x_train, y_train, x_valid, y_valid,
                      params=param_config,
                      job_name=job_name,
                      username=author_name,
                      slack_token=None,
                      log_dir=str(log_dir),
                      id_valid=id_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO(luke): Find a better solution for user config. Right now
    # we have to explicitly define a input field in trainer JS,
    # a state variable, conf calls in etl/train_dag.py, and the
    # definitions here. There may not be a better solution, so this
    # path of commands needs to be well documented.
    parser.add_argument('--data_name',
                        help='The json config, used by the web trainer')
    parser.add_argument('--batch_size',
                        help='The json config, used by the web trainer')
    parser.add_argument('--val_split',
                        help='The json config, used by the web trainer')
    parser.add_argument('--max_epochs',
                        help='The json config, used by the web trainer')
    parser.add_argument('--job_name',
                        help='The json config, used by the web trainer')
    parser.add_argument('--author_name',
                        help='The json config, used by the web trainer')
    parser.add_argument('--three_fold_split')
    args = parser.parse_args()

    run_web_gpu1708_job(args.data_name,
                        args.batch_size,
                        args.val_split,
                        args.max_epochs,
                        args.job_name,
                        args.author_name,
                        args.three_fold_split)
