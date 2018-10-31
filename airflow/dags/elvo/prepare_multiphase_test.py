import logging
import sys

import pytest
from google.cloud import storage

from elvo.prepare_arrays import load_scan, preprocess_scan
from .prepare_multiphase import process_patient

@pytest.mark.skip
def test_process_patient_no_err():
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('data-elvo')
    process_patient('multiphase/negative/N129/', bucket)

@pytest.mark.skip
def test_process_patient_local_no_err(caplog):
    caplog.set_level(logging.DEBUG)
    mip_dir = '/Users/lukezhu/Desktop/research/' \
              'elvo/airflow/dags/tmp/multiphase/negative/N129/mip2/'
    slices = load_scan(mip_dir)
    arrays = preprocess_scan(slices)
