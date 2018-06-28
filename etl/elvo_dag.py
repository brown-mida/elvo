import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from compress_numpy import compress_numpy
from dicom_to_npy import dicom_to_npy

default_args = {
    'owner': 'luke',
    'start_date': datetime(2018, 6, 26, 8),
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

slack_confirmation = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'ELVO data synced',
    dag=dag,
)

dropbox_to_gcs >> elvos_anon_to_numpy_op
elvos_anon_to_numpy_op >> compress_numpy_op
compress_numpy_op >> slack_confirmation
# TODO: Add spreadsheet_to_gcs and encode_labels to the DAG
