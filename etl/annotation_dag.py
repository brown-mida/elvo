"""
Airflow DAG for updating annotations from Google Drive to GCS.
"""

import datetime

import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from roi_drive_to_gcs import roi_to_gcs

# Set outermost parameters
default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 5, 12),
}

# Initialize DAG
# dag_id: DAG name, self explanatory
# description: description of DAG's purpose
# default_args: previously specified params
# catchup: whether or not intervals are automated; set to False to
#         indicate that the DAG will only run for the most current instance
#         of the interval series
# max_active_runs: maximum number of active DAG runs, beyond this
#         number of DAG runs in a running state, the scheduler won't create
#         new active DAG runs
dag = DAG(dag_id='load_annotations',
          description='Load the annotations from Drive to GCS',
          default_args=default_args,
          catchup=False,
          max_active_runs=1)

# Define first operation in DAG (upload ROI annotations to GCS)
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
load_op = PythonOperator(task_id='load',
                         python_callable=roi_to_gcs,
                         dag=dag)

# Define second operation in DAG (notify Slack channel with a token)
# task_id: name of operation, self explanatory
# channel: the Slack channel that will display the notification
# username: the Slack user who will be listed as posting the notification
# token: the Slack token
# text: text of Slack token
# dag: the previously created DAG
notify_slack = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'Uploaded annotations to GCS',
    dag=dag,
)

# Specify order of DAG operations
load_op >> notify_slack
