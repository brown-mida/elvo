"""
Creates a Directed Acyclic Graph (DAG) to upload numpy data and
data labels to Google Cloud Storage using Airflow.
"""

from datetime import datetime

import os
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from compress_numpy import compress_gcs_arrays
from dicom_to_npy import dicom_to_npy
from prepare_labels import prepare_labels
from spreadsheet_to_gcs import spreadsheet_to_gcs

# Set outermost parameters
default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime(2018, 6, 28, 5),
}

# Directory containing compressed DICOM files
ELVOS_ANON = 'ELVOs_anon/'

# Directory to store raw numpy files
RAW_NUMPY = 'airflow/npy/'

# Directory to store npz (compressed numpy) files
COMPRESSED_NUMPY = 'airflow/npz/'

# Create DAG
# dag_id: DAG name, self explanatory
# description: description of DAG's purpose
# default_args: previously specified params
# catchup: whether or not intervals are automated; set to False to
#         indicate that the DAG will only run for the most current instance
#         of the interval series
# max_active_runs: maximum number of active DAG runs, beyond this
#         number of DAG runs in a running state, the scheduler won't create
#         new active DAG runs
dag = DAG(dag_id='elvo_main',
          description='The main DAG for loading ELVO data from Dropbox and'
                      ' and Google Drive',
          default_args=default_args,
          catchup=False,
          max_active_runs=1)

# Define operation in DAG (upload new data in
#       https://www.dropbox.com/home/ELVOs_anon to gs://elvos/elvos_anon)
# task_id: name of operation, self explanatory
# bash_command: the terminal command that will be called by this operation
# dag: the previously created DAG
dropbox_to_gcs = BashOperator(task_id='dropbox_to_gcs',
                              bash_command='python3 /home/lukezhu/'
                                           'elvo-analysis/'
                                           'etl/dropbox_to_gcs.py',
                              dag=dag)

# Define operation in DAG (converts compressed files containing DICOM files
#       to numpy files)
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
elvos_anon_to_numpy_op = PythonOperator(task_id='elvos_anon_to_numpy',
                                        python_callable=lambda: dicom_to_npy(
                                            ELVOS_ANON, RAW_NUMPY),
                                        dag=dag)

# Define operation in DAG (compresses files into .npz files consisting of
#       10 numpy arrays each)
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
compress_numpy_op = PythonOperator(task_id='compress_numpy',
                                   python_callable=lambda: compress_gcs_arrays(
                                       RAW_NUMPY, COMPRESSED_NUMPY),
                                   dag=dag)

# Define operation in DAG (fetches ELVO key spreadsheet from Google Drive,
#       and saves it in Google Cloud Storage.)
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
spreadsheet_to_gcs_op = PythonOperator(task_id='spreadsheet_to_gcs',
                                       python_callable=spreadsheet_to_gcs,
                                       dag=dag)

# Define operation in DAG (creates and loads a labels.csv file to GCS)
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
encode_labels_op = PythonOperator(task_id='encode_labels',
                                  python_callable=prepare_labels,
                                  dag=dag)

# Define operation in DAG (notify Slack channel with a token)
# task_id: name of operation, self explanatory
# channel: the Slack channel that will display the notification
# username: the Slack user who will be listed as posting the notification
# token: the Slack token
# text: text of Slack token
# dag: the previously created DAG
slack_confirmation = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='airflow',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'ELVO data synced',
    dag=dag,
)

# Specify order of DAG operations
dropbox_to_gcs >> elvos_anon_to_numpy_op
spreadsheet_to_gcs_op >> encode_labels_op
elvos_anon_to_numpy_op >> compress_numpy_op
compress_numpy_op >> slack_confirmation
