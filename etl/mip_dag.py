import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

from mip import normal_mip
from multichannel_mip import multichannel_mip
from overlap_mip import overlap_mip

default_args = {
    'owner': 'luke',
    'start_date': datetime(2018, 6, 28, 5),
}

ELVOS_ANON = 'ELVOs_anon/'
RAW_NUMPY = 'airflow/npy/'
COMPRESSED_NUMPY = 'airflow/npz/'

dag = DAG(dag_id='mip_dag', default_args=default_args)

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
    text=f'Coronal and axial scans MIPed',
    dag=dag,
)

normal_mip_op >> multichannel_mip_op >> overlap_mip_op >> slack_confirmation
