import datetime

import matplotlib
import numpy as np
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from google.cloud import storage

matplotlib.use('agg')  # noqa: E402
from matplotlib import pyplot as plt

# A list of subdirectories of 'gs://elvos/processed' to ignore
BLACKLIST = [
    'luke',
    'luke1',
    'mip_three10',
    'mip_three4',
]


def to_public_png(npy_blob: storage.Blob, public_bucket: storage.Bucket):
    """
    Converts the .npy blob into a png file and uploads it to the public
    bucket.

    :param npy_blob:
    :param public_bucket:
    :return:
    """
    npy_filepath = f'/tmp/{npy_blob.name.split("/")[-1]}'
    npy_blob.download_to_filename(npy_filepath)
    arr = np.load(npy_filepath)

    png_filepath = npy_filepath.replace('.npy', '.png')
    plt.imsave(png_filepath, arr)

    png_blob_name = npy_blob.name.replace('.npy', '.png')
    png_blob = public_bucket.blob(png_blob_name)
    png_blob.upload_from_filename(png_filepath)
    os.remove(npy_filepath)
    os.remove(png_filepath)


def upload_numpy_files():
    """
    Uploads all numpy files to gs://elvos-public.

    Only the extension of the name will be changed. For example
    gs://elvos/processed/processed-lower/abc.npy will be uploaded
    as gs://elvos-public/processed/processed-lower/abc.png

    Blacklisted folders will be ignored.

    :return:
    """
    gcs_client = storage.Client(project='elvo-198322')
    in_bucket = gcs_client.get_bucket('elvos')
    out_bucket = gcs_client.get_bucket('elvos-public')

    blob: storage.Blob
    for blob in in_bucket.list_blobs(prefix='processed/'):
        to_ignore = False
        for s in BLACKLIST:
            if blob.name.split('/')[1] == s:
                print(f'ignoring blacklisted blob {blob.name}')
                to_ignore = True
                break

        if not to_ignore and blob.name.endswith('.npy'):
            print(f'uploading blob {blob.name}')
            to_public_png(blob, out_bucket)


default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 24),
}

dag = DAG(dag_id='upload_processed_data',
          description='Uploads processed numpy array data as pngs for the'
                      ' web app',
          default_args=default_args,
          catchup=False)

upload_numpy_files_op = PythonOperator(task_id='upload_numpy_files',
                                       python_callable=upload_numpy_files,
                                       dag=dag)
