"""
Creates a Directed Acyclic Graph (DAG) to upload ROI annotations to
Google Cloud Storage using Airflow.
"""

import datetime

import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from roi_drive_to_gcs import roi_to_gcs

# Set metadata parameters
default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 5, 12),
}

# Create DAG
# default_args: previously specified params
# catchup:
# max_active_runs:
dag = DAG(dag_id='load_annotations',
          description='Load the annotations from Drive to GCS',
          default_args=default_args,
          catchup=False,
          max_active_runs=1)

# First operation in DAG-- upload ROI annotations to GCS
load_op = PythonOperator(task_id='load',
                         python_callable=roi_to_gcs,
                         dag=dag)

# Second operation in DAG-- notify Slack channel with a token
notify_slack = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'Uploaded annotations to GCS',
    dag=dag,
)

# Run DAG operations
load_op >> notify_slack
