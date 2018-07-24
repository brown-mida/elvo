import datetime
import logging

import paramiko
from airflow import DAG
from airflow.models import BaseOperator
from airflow.operators.sensors import BaseSensorOperator


def run_bluenot(config: dict):
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
            " nohup python3 ml/bluenot.py"
            # TODO(luke): Escape/validate user input (if necessary)
            f" --data_name={config['dataName']}"
            f" --max_epochs={config['maxEpochs']}"
            f" --job_name={config['jobName']}"
            f" --author_name={config['authorName']}"
            f" --batch_size={config['batchSize']}"
            f" --val_split={config['valSplit']}"
            " > web-trainer.log 2>&1 & echo $!'"
        )
        err = stderr.read()
        if err != b'':
            raise ValueError(f'stderr contains message: {err}')
        pid = int(stdout.read())
    finally:
        client.close()
    return pid


class RunBluenotOperator(BaseOperator):
    def execute(self, context):
        conf = context['dag_run'].conf
        logging.info('configuration is: {}'.format(conf))
        run_bluenot(conf)


def count_processes_matching(fragment: str):
    """
    Returns the number of processes on gpu1708 containing the fragment.

    This is equivalent to 'pgrep -f {fragment} | wc -l'.

    :param fragment: the string to match with 'pgrep'
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
        self.retries = 5
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

run_bluenot_op = RunBluenotOperator(task_id='run_bluenot',
                                    dag=train_dag)

sense_complete_op = WebTrainerSensor(task_id='sense_complete',
                                     fragment='data_name',
                                     dag=train_dag,
                                     retries=3)

run_bluenot_op >> sense_complete_op
