import os
import time

import paramiko
import pytest

from train_dag import run_bluenot, count_processes_matching


def teardown_module():
    # Make sure that no gpu processes are left behind.
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('ssh.cs.brown.edu', username='lzhu7', password='')
    client.exec_command(
        "ssh gpu1708 'pgrep -f config_luke | xargs kill -9'")


@pytest.mark.skipif(os.uname().nodename != 'airflow',
                    reason='Requires correct ssh configuration')
def test_run_bluenot_return():
    pid = run_bluenot()
    assert isinstance(pid, int)


@pytest.mark.skipif(os.uname().nodename != 'airflow',
                    reason='Requires correct ssh configuration')
def test_run_bluenot_creates_process():
    run_bluenot()

    time.sleep(3)  # Sleep so the command can be propagated.

    assert count_processes_matching('config_luke') > 0
