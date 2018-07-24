"""
Connection logic with Google Cloud Storage.
"""
import logging
import pathlib
import subprocess

import keras
import numpy as np
import os
import warnings
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage

from blueno.elasticsearch import JOB_INDEX
from blueno.slack import _create_all_plots


def equal_array_counts(arrays_dir: pathlib.Path,
                       arrays_gsurl: str):
    if 'elvos' not in arrays_gsurl:
        raise ValueError('Expected elvos in URL')

    # +1 to avoid leading /
    prefix_i = arrays_gsurl.find('elvos') + len('elvos') + 1
    try:
        client = storage.Client(project='elvo-198322')
    except DefaultCredentialsError:
        warnings.warn('Set GOOGLE_APPLICATION_CREDENTIALS in your config '
                      'file')
        client = storage.Client.from_service_account_json(
            '/gpfs/main/home/lzhu7/elvo-analysis/secrets/'
            'elvo-7136c1299dea.json',
        )

    bucket = client.get_bucket('elvos')
    gcs_count = len(list(bucket.list_blobs(prefix=arrays_gsurl[prefix_i:])))
    local_count = len([0 for _ in arrays_dir.iterdir()])

    return local_count == gcs_count


def fetch_model(service_account_path=None, save_path=None, **kwargs):
    """
    Downloads the relevant model from Google Cloud, given specific
    parameters of the model.

    e.g. fetch_model(_id='A8U2pGQBLx2QGQijrVch') will download the
        model with the specific id.

    :param service_account_path: The path of the GCS service account JSON.
    :param save_path: The path to save the model.
    :param kwargs: The series of parameters to specify the model.
    :return:
    :raise ValueError: If the parameter query returns no result, or
        returns more than 1 result.
    """
    if service_account_path is None:
        service_account_path = '../credentials/client_secret.json'

    if save_path is None:
        save_path = '../tmp/downloaded_models'

    params = kwargs
    matches = JOB_INDEX.search()
    for param in params.keys():
        d = {param: params[param]}
        matches = matches.query('match', **d)
    response = matches.execute()

    if len(response) > 1:
        raise ValueError(('Query not specific enough. '
                          'Found {} results'.format(len(response))))
    elif len(response) == 0:
        raise ValueError('Found 0 results.')

    result = list(response)[0]
    result_str = '{}-{}'.format(result.job_name, result.created_at)

    gcs_client = storage.Client.from_service_account_json(
        service_account_path
    )
    bucket = gcs_client.get_bucket('elvos')
    blob = storage.Blob('models/{}'.format(result_str), bucket)
    blob.download_to_filename(
        '{}/{}.hdf5'.format(save_path, result_str)
    )


def upload_model_to_gcs(job_name, created_at, model_filepath):
    """Uploads the model at the given filepath to
    gs://elvos/sorted_models/{job_name}-{created_at}.hdf5
    """
    gcs_filepath = 'gs://elvos/sorted_models/{}-{}.hdf5'.format(
        # Remove the extension
        job_name,
        created_at,
    )
    # Do not change, this is log is used to get the gcs link
    logging.info('uploading model {} to {}'.format(
        model_filepath,
        gcs_filepath,
    ))

    try:
        subprocess.run(
            ['/bin/bash',
             '-c',
             'gsutil cp {} {}'.format(model_filepath, gcs_filepath)],
            check=True)
    except subprocess.CalledProcessError:
        # gpu1708 specific code
        subprocess.run(
            ['/bin/bash',
             '-c',
             '/gpfs/main/home/lzhu7/google-cloud-sdk/bin/'
             'gsutil cp {} {}'.format(model_filepath, gcs_filepath)],
            check=True)


def upload_gcs_plots(x_train: np.ndarray,
                     x_valid: np.ndarray,
                     y_valid: np.ndarray,
                     model: keras.models.Model,
                     history: keras.callbacks.History,
                     job_name: str,
                     created_at: str,
                     id_valid: np.ndarray = None,
                     chunk: bool = False,
                     plot_dir: pathlib.Path = pathlib.Path('/tmp')):
    """
    Uploads a loss graph, accuracy, and confusion matrix plots in addition
    to useful data about the model to gcs.

    Saves to gs://elvos-public/plots/{job_name}-{created_at}/

    :param x_train:
    :param x_valid:
    :param y_valid:
    :param model:
    :param history:
    :param job_name:
    :param params:
    :param token:
    :param id_valid:
    :param chunk:
    :param plot_dir:
    :return:
    """
    os.makedirs(str(plot_dir), exist_ok=True)
    loss_path = pathlib.Path(plot_dir) / 'loss.png'
    acc_path = pathlib.Path(plot_dir) / 'acc.png'
    cm_path = pathlib.Path(plot_dir) / 'cm.png'
    tp_path = pathlib.Path(plot_dir) / 'true_positives.png'
    fp_path = pathlib.Path(plot_dir) / 'false_positives.png'
    tn_path = pathlib.Path(plot_dir) / 'true_negatives.png'
    fn_path = pathlib.Path(plot_dir) / 'false_negatives.png'

    _create_all_plots(x_train, x_valid, y_valid, model, history,
                      loss_path, acc_path, cm_path, tn_path, tp_path,
                      fn_path, fp_path, chunk, id_valid)
    client = storage.Client(project='elvo-198322')
    bucket = client.bucket('elvos-public')
    for filename in os.listdir(str(plot_dir)):
        blob = bucket.blob(f'plots/{job_name}-{created_at}/{filename}')
        blob.upload_from_filename(str(plot_dir / filename))
