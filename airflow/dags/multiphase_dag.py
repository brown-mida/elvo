import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from elvo.prepare_multiphase import prepare_multiphase

default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 10, 29),
}

dag = DAG(dag_id='elvo_multiphase',
          description='Prepares multiphase ELVOs',
          schedule_interval=None,
          default_args=default_args,
          catchup=False,
          max_active_runs=1)

op = PythonOperator(python_callable=prepare_multiphase,
                    dag=dag,
                    task_id='prepare_arrays')
