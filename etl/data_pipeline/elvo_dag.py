import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from data_pipeline.compress_numpy import compress_numpy
from data_pipeline.dicom_to_npy import dicom_to_npy
from data_pipeline.encode_labels import encode_labels
from data_pipeline.spreadsheet_to_gcs import spreadsheet_to_gcs

default_args = {
    'owner': 'luke',
    'start_date': datetime(2018, 6, 28, 5),
}

ELVOS_ANON = 'ELVOs_anon/'
RAW_NUMPY = 'airflow/npy/'
COMPRESSED_NUMPY = 'airflow/npz/'

dag = DAG(dag_id='elvo_main', default_args=default_args)

dropbox_to_gcs = BashOperator(task_id='dropbox_to_gcs',
                              bash_command='python3 /home/lukezhu/'
                                           'elvo-analysis/'
                                           'etl/dropbox_to_gcs.py',
                              dag=dag)

elvos_anon_to_numpy_op = PythonOperator(task_id='elvos_anon_to_numpy',
                                        python_callable=lambda: dicom_to_npy(
                                            ELVOS_ANON, RAW_NUMPY),
                                        dag=dag)

compress_numpy_op = PythonOperator(task_id='compress_numpy',
                                   python_callable=lambda: compress_numpy(
                                       RAW_NUMPY, COMPRESSED_NUMPY),
                                   dag=dag)

spreadsheet_to_gcs_op = PythonOperator(task_id='spreadsheet_to_gcs',
                                       python_callable=spreadsheet_to_gcs,
                                       dag=dag)

encode_labels_op = PythonOperator(task_id='encode_labels',
                                  python_callable=encode_labels,
                                  dag=dag)

slack_confirmation = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'ELVO data synced',
    dag=dag,
)

dropbox_to_gcs >> elvos_anon_to_numpy_op
spreadsheet_to_gcs_op >> encode_labels_op
elvos_anon_to_numpy_op >> compress_numpy_op
compress_numpy_op >> slack_confirmation
