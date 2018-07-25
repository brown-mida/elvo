"""
The script called by the web trainer.

Do not change this filename without updating train_dag as well
"""
import argparse
import logging
import pathlib

import keras

import blueno
import bluenot
import generators.luke
import models.luke


def run_web_gpu1708_job(data_name: str,
                        batch_size: int,
                        val_split: float,
                        max_epochs: int,
                        job_name: str,
                        author_name: str):
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
        max_epochs=int(max_epochs),
        job_name=job_name,
    )

    logging.info('training web job {}'.format(param_config))

    bluenot.hyperoptimize(
        [param_config],
        author_name,
        num_gpus=1,
        gpu_offset=3,
        log_dir=str(log_dir),
    )


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
    args = parser.parse_args()

    run_web_gpu1708_job(args.data_name,
                        args.batch_size,
                        args.val_split,
                        args.max_epochs,
                        args.job_name,
                        args.author_name)
