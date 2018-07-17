import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from roi_preprocess import run_preprocess
from transform_positives import run_transform

default_args = {
    'owner': 'hal',
    'start_date': datetime(2018, 7, 17, 5),
}

dag = DAG(dag_id='roi_chunk_dag', default_args=default_args)

preprocess_op = PythonOperator(task_id='run_preprocess',
                               python_callable=run_preprocess(),
                               dag=dag)

transform_op = PythonOperator(task_id='run_transform',
                              python_callable=run_transform(),
                              dag=dag)

slack_confirmation = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text=f'ROI chunk data processed + uploaded',
    dag=dag,
)

preprocess_op >> transform_op >> slack_confirmation
