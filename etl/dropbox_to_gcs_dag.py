import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

default_args = {
    'owner': 'luke',
    'start_date': datetime(2018, 6, 26),
}

dag = DAG(dag_id='dropbox_to_gcs', default_args=default_args)

op = BashOperator(task_id='dropbox_to_gcs_op',
                  bash_command='python3 /home/lukezhu/elvo-analysis/etl/dropbox_to_gcs.py',
                  dag=dag)

notify_slack = SlackAPIPostOperator(
    task_id='notify_slack',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'Dropbox to GCS workflow run. '
         f'Check the Airflow UI to see the results',
    dag=dag,
)

op >> notify_slack
