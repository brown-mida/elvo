import datetime

import paramiko
import requests
from airflow import DAG
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.sensors import BaseSensorOperator, HttpSensor


def run_bluenot():
    """
    Runs blunot.py on gpu1708 and returns it's PID.

    # TODO: User-defined config (new data, params, etc.)
    # TODO: How about downloading new data.
    # TODO: How about queueing jobs?
    :return:
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    try:
        client.connect('ssh.cs.brown.edu', username='lzhu7', password='')
        stdin, stdout, stderr = client.exec_command(
            "ssh gpu1708 'cd elvo-analysis;"
            " source venv/bin/activate;"
            " nohup python3 ml/bluenot.py --config=config_luke"
            " > /dev/null 2>&1 & echo $!'"
        )
        err = stderr.read()
        if err != b'':
            raise ValueError(f'stderr contains message: {err}')
        pid = int(stdout.read())
    finally:
        client.close()
    return pid


def count_processes_matching(fragment: str):
    """
    Returns the number of processes on gpu1708 containing the fragment.

    This is equivalent to 'pgrep -f {fragment} | wc -l'.

    :param config_luke:
    :return:
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    try:
        client.connect('ssh.cs.brown.edu', username='lzhu7', password='')
        stdin, stdout, stderr = client.exec_command(
            f"ssh gpu1708 'pgrep -f {fragment} | wc -l'"
        )
        count = int(stdout.read()) - 1  # The command above is counted
    finally:
        client.close()
    return count


class WebTrainerSensor(BaseSensorOperator):
    def __init__(self, fragment, *args, **kwargs):
        self.fragment = fragment
        super().__init__(*args, **kwargs)

    def poke(self, context):
        """
        Returns true if no process matching the fragment is found.

        :param context:
        :return:
        """
        if count_processes_matching(self.fragment) == 0:
            return True
        return False


args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 20, 2),
}

train_dag_id = 'train_model'
train_dag = DAG(dag_id='train_model',
                description='Trains a ML model on gpu1708',
                default_args=args,
                schedule_interval=None,
                max_active_runs=1)

run_bluenot_op = PythonOperator(task_id='run_bluenot',
                                python_callable=run_bluenot,
                                dag=train_dag)

sense_complete_op = WebTrainerSensor(task_id='sense_complete',
                                     fragment='config_luke',
                                     dag=train_dag)

run_bluenot_op >> sense_complete_op

trigger_dag_id = 'trigger_model'
trigger_dag = DAG(dag_id=trigger_dag_id,
                  default_args=args,
                  schedule_interval=datetime.timedelta(minutes=2),
                  catchup=False,
                  max_active_runs=1)


def check_fn(response: requests.Response):
    return response.json()['is_job']


# This isn't the best solution since GET assumes idempotentency
start_sensor = HttpSensor(endpoint='model/pop',
                          task_id='sense_trigger',
                          http_conn_id='flask',
                          response_check=check_fn,
                          dag=trigger_dag)


def trigger_fn(context, dag_run_object):
    return dag_run_object


trigger_bluenot = TriggerDagRunOperator(task_id='trigger_bluenot',
                                        trigger_dag_id=train_dag_id,
                                        python_callable=trigger_fn,
                                        dag=trigger_dag)

start_sensor >> trigger_dag
