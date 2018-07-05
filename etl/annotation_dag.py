import datetime
import os

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from roi_drive_to_gcs import roi_to_gcs

default_args = {
    'owner': 'luke',
    'start_date': datetime.datetime(2018, 7, 5, 12),
}

dag = DAG(dag_id='load_annotations',
          description='Load the annotations from Drive to GCS')

load_op = PythonOperator(task_id='load',
                         python_callable=roi_to_gcs,
                         dag=dag)

notify_slack = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'Uploaded annotations to GCS',
    dag=dag,
)

load_op >> notify_slack
