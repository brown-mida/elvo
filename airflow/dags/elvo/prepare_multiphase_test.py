from google.cloud import storage

from .prepare_multiphase import process_patient


def test_process_patient_no_err():
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('data-elvo')
    process_patient('multiphase/negative/N129/', bucket)
