"""
Airflow DAG for keeping MIPs on GCS updated.
"""

from datetime import datetime

import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from axial_to_coronal_and_sagittal import axial_to_coronal_and_sagittal
from mip import normal_mip
from multichannel_mip import multichannel_mip
from overlap_mip import overlap_mip

# Set outermost parameters
default_args = {
    'owner': 'hal',
    'start_date': datetime(2018, 6, 28, 5),
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
dag = DAG(dag_id='mip_dag', default_args=default_args)

# Define operations:
axial_to_coronal_and_sagittal_op = \
    PythonOperator(task_id='axial_to_coronal_and_sagittal',
                   python_callable=axial_to_coronal_and_sagittal,
                   dag=dag)

normal_mip_op = PythonOperator(task_id='normal_mip',
                               python_callable=normal_mip,
                               dag=dag)

multichannel_mip_op = PythonOperator(task_id='multichannel_mip',
                                     python_callable=multichannel_mip,
                                     dag=dag)

overlap_mip_op = PythonOperator(task_id='overlap_mip',
                                python_callable=overlap_mip,
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
    text='Coronal and axial scans MIPed',
    dag=dag,
)

# Specify order of DAG operations
axial_to_coronal_and_sagittal_op >> normal_mip_op >> \
    multichannel_mip_op >> overlap_mip_op >> slack_confirmation
