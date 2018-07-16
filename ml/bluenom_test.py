import pathlib

import os
import pytest
from elasticsearch_dsl import connections

import bluenom
from blueno.elasticsearch import JOB_INDEX


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses data only on gpu1708')
def test_bluenom_idempotent():
    connections.create_connection(hosts=['http://104.196.51.205'])
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs')
    bluenom.bluenom(path, gpu1708=True)
    first_run_count = JOB_INDEX.search().count()
    bluenom.bluenom(path, gpu1708=True)
    second_run_count = JOB_INDEX.search().count()
    connections.remove_connection('default')
    assert first_run_count == second_run_count
