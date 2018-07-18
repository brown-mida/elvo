import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from mip_with_segmentation import normal_mip
from multichannel_mip_with_segmentation import multichannel_mip
from overlap_mip_with_segmentation import overlap_mip

default_args = {
    'owner': 'amy',
    'start_date': datetime(2018, 6, 28, 5),
}

dag = DAG(dag_id='segmented_mip_dag', default_args=default_args)


normal_mip_op = PythonOperator(task_id='normal_mip',
                               python_callable=normal_mip,
                               dag=dag)

multichannel_mip_op = PythonOperator(task_id='multichannel_mip',
                                     python_callable=multichannel_mip,
                                     dag=dag)

overlap_mip_op = PythonOperator(task_id='overlap_mip',
                                python_callable=overlap_mip,
                                dag=dag)

slack_confirmation = SlackAPIPostOperator(
    task_id='slack_confirmation',
    channel='i-utra',
    username='airflow',
    token=os.environ['SLACK_TOKEN'],
    text='Coronal and axial scans stripped and MIPed',
    dag=dag,
)

normal_mip_op >> multichannel_mip_op >> overlap_mip_op >> slack_confirmation
