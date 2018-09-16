"""
Creates a Directed Acyclic Graph (DAG) to upload skull-stripped MIPed data
to Google Cloud Storage using Airflow.
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from mip_with_segmentation import normal_mip
from multichannel_mip_with_segmentation import multichannel_mip
from overlap_mip_with_segmentation import overlap_mip

# Set outermost parameters
default_args = {
    'owner': 'amy',
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
dag = DAG(dag_id='segmented_mip_dag', default_args=default_args)

# Define operation in DAG (generate axial and coronal MIPs, strip the skull,
#       upload to gs://elvos/stripped_mip_data/numpy/normal/{axial or coronal})
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
normal_mip_op = PythonOperator(task_id='normal_mip',
                               python_callable=normal_mip,
                               dag=dag)

# Define operation in DAG (generate 3-channel MIPs, strip the skull, upload
#        to gs://elvos/stripped_mip_data/numpy/multichannel/{axial or coronal})
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
multichannel_mip_op = PythonOperator(task_id='multichannel_mip',
                                     python_callable=multichannel_mip,
                                     dag=dag)

# Define operation in DAG (get 20-channel overlapping MIPs, strip the skull
#      upload to gs://elvos/stripped_mip_data/numpy/overlap/{axial or coronal})
# task_id: name of operation, self explanatory
# python_callable: the imported method that will be called by this operation
# dag: the previously created DAG
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
    channel='airflow',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text='Coronal and axial scans stripped and MIPed',
    dag=dag,
)

# Specify order of DAG operations
normal_mip_op >> multichannel_mip_op >> overlap_mip_op >> slack_confirmation
