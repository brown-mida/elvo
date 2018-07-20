import datetime

import paramiko
from airflow import DAG
from airflow.operators.python_operator import PythonOperator


def run_bluenot():
    """
    Runs blunot.py on gpu1708.

    # TODO: User-defined config (new data, params, etc.)
    # TODO: How about downloading new data.
    # TODO: How about queueing jobs?
    :return:
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('ssh.cs.brown.edu', username='lzhu7', password='')
    client.exec_command(
        "ssh gpu1708 'cd elvo-analysis "
        "&& source venv/bin/activate "
        "&& nohup python3 ml/bluenot.py "
        "--config=config_luke > /dev/null 2>&1 &'")
    client.close()


args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 20, 2),
}

train_dag_id = 'train_model'
train_dag = DAG(dag_id='train_model',
                description='Trains a ML model on gpu1708',
                default_args=args,
                schedule_interval=None)
# TODO: Figure out how to only run 1 model at a time
run_bluenot_op = PythonOperator(task_id='run_bluenot',
                                python_callable=run_bluenot,
                                dag=train_dag)

# trigger_dag_id = 'trigger_model'
# trigger_dag = DAG(dag_id=trigger_dag_id, default_args=args, schedule_interval=None)
# start_sensor = HttpSensor(task_id='start',
#                           http_conn_id='??',
#                           params='??',
#                           dag=trigger_dag)
# trigger_bluenot = TriggerDagRunOperator(task_id='trigger_bluenot',
#                                         trigger_dag_id=train_dag_id,
#                                         dag=trigger_dag)
#
# start_sensor >> trigger_dag
