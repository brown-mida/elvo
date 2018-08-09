"""
Airflow DAG to keep chunks on GCS up to date.
"""
from datetime import datetime

import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from roi_preprocess import run_preprocess
from transform_positives import run_transform

# Set outermost parameters
default_args = {
    'owner': 'hal',
    'start_date': datetime(2018, 7, 17, 5),
}

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
dag = DAG(dag_id='roi_chunk_dag', default_args=default_args)

# Define operation in DAG (preprocess images from airflow/npy by converting to
#       chunks and saving in gs://elvos/chunk_data/{type}/{elvo_status}/, also
#       update labels and save as annotated_labels.csv)
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
preprocess_op = PythonOperator(task_id='run_preprocess',
                               python_callable=run_preprocess,
                               dag=dag)

# Define operation in DAG (transform positive cubes via 24 lossless rotations
#       and reflections and upload to gs://elvos/chunk_data/{type}/positives/;
#       also update labels and save as augmented_annotated_labels.csv)
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
transform_op = PythonOperator(task_id='run_transform',
                              python_callable=run_transform,
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
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'ROI chunk data processed + uploaded',
    dag=dag,
)

# Specify order of DAG operations
preprocess_op >> transform_op >> slack_confirmation
