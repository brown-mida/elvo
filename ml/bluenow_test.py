import os

import pytest

from bluenow import run_web_gpu1708_job


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Code only runs on gpu1708')
def test_run_web_gpu1708_job_three_fold():
    job_name = 'test-web-job'
    run_web_gpu1708_job('processed-lower', 8, 0.1, 1, job_name,
                        'test-author', True)
    log_dir = '/home/lzhu7/elvo-analysis/logs'

    for filename in os.listdir(log_dir):
        if filename.startswith(job_name):
            with open(log_dir + '/' + filename) as f:
                content = f.read()
                assert 'error' not in content.lower()
