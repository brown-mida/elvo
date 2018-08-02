import numpy as np
from google.cloud import storage

from preprocess3d_dag import process_patient, load_split_data


def test_process_patient_length():
    arr = np.zeros(shape=(300, 300, 300))
    chunks = process_patient('abc', arr, [])
    assert len(chunks) == int(300 / 32) ** 3


def test_process_patient_centers():
    arr = np.zeros(shape=(300, 300, 300))
    centers = [(32, 32, 32), (170, 170, 170)]
    chunks = process_patient('abc', arr, centers)
    positives = sum([1 for c in chunks if c.label == 1])
    assert positives == 2


def test_process_patient_centers_same_chunk():
    arr = np.zeros(shape=(300, 300, 300))
    centers = [(32, 32, 32), (50, 50, 50)]
    chunks = process_patient('abc', arr, centers)
    positives = sum([1 for c in chunks if c.label == 1])
    assert positives == 1


def test_load_split_data():
    client = storage.Client(project='elvo-198322')
    bucket = client.bucket('elvos')
    test_blob_name = 'processed3d/airflow-2/arrays/9PMFHK7Q3UGW0UM.npz'
    test_blob = bucket.get_blob(test_blob_name)
    test_labels = {
        '9PMFHK7Q3UGW0UM': {
            '32-32-32': 1,
            '32-32-64': 0,
        }
    }
    x, y = load_split_data([test_blob], test_labels)
    assert x.shape == (2, 32, 32, 32)
    assert y.shape == (2,)
