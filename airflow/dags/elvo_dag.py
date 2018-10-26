from datetime import datetime

import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from dags.elvo.dropbox_to_gcs import dropbox_to_gcs
from dags.elvo.compress_arrays import compress_arrays
from dags.elvo.prepare_arrays import prepare_arrays
from dags.elvo.prepare_labels import prepare_labels
from dags.elvo.spreadsheet_to_gcs import spreadsheet_to_gcs

default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime(2018, 6, 28, 5),
}

ELVOS_ANON = 'ELVOs_anon/'  # Holds compressed DICOM files
RAW_NUMPY = 'airflow/npy/'
COMPRESSED_NUMPY = 'airflow/npz/'

dag = DAG(dag_id='elvo_main',
          description='Loads ELVOs from Dropbox and'
                      ' and labels from Google Drive',
          default_args=default_args,
          catchup=False,
          max_active_runs=1)

dropbox_to_gcs_op = PythonOperator(task_id='dropbox_to_gcs',
                                python_callable=lambda: dropbox_to_gcs(),
                                dag=dag)

elvos_anon_to_numpy_op = PythonOperator(task_id='prepare_arrays',
                                        python_callable=lambda: prepare_arrays(
                                            ELVOS_ANON, RAW_NUMPY),
                                        dag=dag)

compress_numpy_op = PythonOperator(task_id='compress_arrays',
                                   python_callable=lambda: compress_arrays(
                                       RAW_NUMPY, COMPRESSED_NUMPY),
                                   dag=dag)

spreadsheet_to_gcs_op = PythonOperator(task_id='spreadsheet_to_gcs',
                                       python_callable=spreadsheet_to_gcs,
                                       dag=dag)

encode_labels_op = PythonOperator(task_id='prepare_labels',
                                  python_callable=lambda: prepare_labels(
                                      ELVOS_ANON),
                                  dag=dag)

slack_confirmation = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='airflow',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'ELVO data synced',
    dag=dag,
)

dropbox_to_gcs_op >> elvos_anon_to_numpy_op
spreadsheet_to_gcs_op >> encode_labels_op
elvos_anon_to_numpy_op >> compress_numpy_op
compress_numpy_op >> slack_confirmation
