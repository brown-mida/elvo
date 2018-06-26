import io
import logging
import os
from datetime import datetime
from typing import Dict

import numpy as np
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator
from google.cloud import storage


def save_arrays(arrays: Dict[str, np.ndarray],
                filename: str,
                bucket: storage.Bucket):
    out_stream = io.BytesIO()
    np.savez_compressed(out_stream, **arrays)
    out_stream.seek(0)
    out_blob = bucket.blob(filename)
    out_blob.upload_from_file(out_stream)


def compress_files():
    """Saves the arrays in batches of 10 in
    gs://elvos/numpy_compressed/.
    """
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('elvos')

    blob: storage.Blob
    arrays = {}
    i = 0
    for blob in bucket.list_blobs(prefix='numpy/'):
        patient_id = blob.name[len('numpy/'): -len('.npy')]
        if len(arrays) > 10:
            logging.info(f'uploading arrays: {list(arrays.keys())}')
            save_arrays(arrays,
                        'numpy_compressed/' + str(i),
                        bucket)
            arrays = {}
            i += 1
        in_stream = io.BytesIO()
        logging.info(f'downloading {blob.name}, patient id: {patient_id}')
        blob.download_to_file(in_stream)
        in_stream.seek(0)
        arr = np.load(in_stream)
        arrays[patient_id] = arr

    if len(arrays) > 0:  # Upload remaining files
        logging.info(f'uploading arrays: {list(arrays.keys())}')
        save_arrays(arrays,
                    'numpy_compressed/' + str(i),
                    bucket)


default_args = {
    'owner': 'luke',
    'start_date': datetime(2018, 6, 26, 8),
}

dag = DAG(dag_id='compress_numpy', default_args=default_args)

op = PythonOperator(task_id='dicom_to_numpy_op',
                    python_callable=compress_files,
                    dag=dag)

notify_slack = SlackAPIPostOperator(
    task_id='notify_slack',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'gs://elvos/numpy to gs://elvos/numpy_compress workflow finished'
         f' running. Check the http://104.196.51.205:8080/ to see the results',
    dag=dag,
)

op >> notify_slack
