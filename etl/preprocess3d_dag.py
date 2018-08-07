"""
Code for a DAG that generates 3D chunks. Eventually to be used in the web
app.
"""
import datetime
import io
import logging
import os
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from dataclasses import dataclass
from google.cloud import storage


@dataclass
class Chunk:
    patient_id: str
    position: Tuple[int, int, int]
    array: np.ndarray
    label: int  # 0 or 1


def preprocess3d(data_name: str):
    """
    Uploads all 32x32x32 chunks to gs://elvos/processed3d. This will be used
    on the web application.

    :param data_name:
    :return:
    """
    client = storage.Client(project='elvo-198322')
    data_bucket = client.bucket('elvos')
    # TODO: Upload sample images along the way
    # image_bucket = client.bucket('elvos-public')

    csv_blob = data_bucket.get_blob('airflow/annotations.csv')
    logging.info(f'loading annotations from {csv_blob.name}')

    csv_filepath = '/tmp/annotations.csv'
    csv_blob.download_to_filename(csv_filepath)
    annotations = pd.read_csv(csv_filepath)
    os.remove(csv_filepath)

    for input_blob in data_bucket.list_blobs(prefix=f'airflow/npy'):
        chunks = process_blob(input_blob, annotations)
        upload_chunks_npz(chunks, data_name, data_bucket)
        upload_labels(chunks, data_name, data_bucket)


def process_blob(input_blob: storage.Blob,
                 annotations: pd.DataFrame) -> List[Chunk]:
    """Takes in a input blob and returns a list of Chunk objects
    extracted from the blob.
    """
    input_filename = input_blob.name.split('/')[-1]
    patient_id = input_filename[:-len('.npy')]
    logging.info(f'processing patient {patient_id}')

    matches = annotations[annotations['patient_id'] == patient_id]
    arr = _download_arr(input_blob)

    centers = []
    for _, row in matches.iterrows():
        center_i = arr.shape[0] - int((row['blue1'] + row['blue2']) // 2)
        center_j = int((row['green1'] + row['green2']) // 2)
        center_k = int((row['red1'] + row['red2']) // 2)
        centers.append((center_i, center_j, center_k))

    logging.info('centers: {}'.format(centers))
    chunks = process_patient(patient_id, arr, centers=centers)
    return chunks


def process_patient(patient_id: str,
                    arr: np.ndarray,
                    centers: List[Tuple[int, int, int]]) -> List[Chunk]:
    if arr.ndim != 3:
        raise ValueError('Array must have 3 dimensions')

    chunks = []
    for i in range(0, arr.shape[0] - arr.shape[0] % 32, 32):
        for j in range(0, arr.shape[1] - arr.shape[1] % 32, 32):
            for k in range(0, arr.shape[2] - arr.shape[2] % 32, 32):
                for center in centers:
                    if (i <= center[0] < i + 32
                            and j <= center[1] < j + 32
                            and k <= center[2] < k + 32):
                        chunk_arr = arr[i: i + 32, j: j + 32, k: k + 32]
                        chunk = Chunk(patient_id=patient_id,
                                      position=(i, j, k),
                                      array=chunk_arr,
                                      label=1)
                        chunks.append(chunk)
                        logging.info('positive chunk: {}'.format(chunk))
                        break
                else:  # The above loop didn't break
                    chunk_arr = arr[i: i + 32, j: j + 32, k: k + 32]
                    chunks.append(Chunk(patient_id=patient_id,
                                        position=(i, j, k),
                                        array=chunk_arr,
                                        label=0))

    return chunks


def _download_arr(input_blob: storage.Blob):
    logging.info(f'downloading numpy file: {input_blob.name}')
    input_stream = io.BytesIO()
    input_blob.download_to_file(input_stream)
    input_stream.seek(0)
    arr = np.load(input_stream)
    input_stream.close()
    return arr


def upload_chunks_npz(chunks: List[Chunk],
                      data_name: str,
                      data_bucket: storage.Bucket) -> None:
    """
    Uploads chunks to gs://elvos/processed3d/{data_name}/arrays/
    as a .npz file ({patient_id}.npz). The NPZ files contain the position
    tuple as a key and the array as the value.

    Example:

    The chunk with position (64, 96, 128) will be saved as:
        '64-96-128': array(...

    :param chunks: a list of chunks with the same patient id.
    :param data_name:
    :param data_bucket:
    :return:
    """
    patient_id = chunks[0].patient_id
    blob = data_bucket.blob(
        f'processed3d/{data_name}/arrays/{patient_id}.npz')
    chunk_dict = {}
    for c in chunks:
        position_str = '-'.join([str(coord) for coord in c.position])
        chunk_dict[position_str] = c.array

    stream = io.BytesIO()
    np.savez(stream, **chunk_dict)
    stream.seek(0)

    logging.info(f'uploading npz file: {blob.name}')
    blob.upload_from_file(stream)
    stream.close()


def upload_labels(chunks: List[Chunk],
                  data_name: str,
                  data_bucket: storage.Bucket) -> None:
    """
    Uploads the labels for a single chunk to
    gs://elvos/processed3d/{data_name}/labels/ as a
    csv file named {patient_id}.csv.

    Note that the position tuple is converted into a hyphen-delimited
    string as in upload_chunks_npz

    :param chunks:
    :param data_name:
    :param data_bucket:
    :return:
    """
    patient_id = chunks[0].patient_id
    labels = []
    for chunk in chunks:
        position_str = '-'.join([str(coord) for coord in chunk.position])
        labels.append((chunk.patient_id, position_str, chunk.label))

    df = pd.DataFrame(data=labels,
                      columns=['patient_id', 'position', 'label'])
    tmp_filepath = f'/tmp/labels-{patient_id}.csv'
    df.to_csv(tmp_filepath, index_label=False)

    blob = data_bucket.blob(f'processed3d/{data_name}/labels/{patient_id}.csv')
    logging.info(f'uploading csv file: {blob.name}')
    blob.upload_from_filename(tmp_filepath)
    os.remove(tmp_filepath)


def save_as_split_dirs(data_name: str) -> None:
    """
    Saves the data as downsampled x_train/x_valid/y_train/y_valid
    files in gs://elvos/processed3d/<data_name>. The files will
    be:
        - f'processed3d/{data_name}/x_train.npy
        - f'processed3d/{data_name}/y_train.npy
        - f'processed3d/{data_name}/x_valid.npy
        - f'processed3d/{data_name}/y_valid.npy

    :param data_name:
    :return:
    """
    client = storage.Client(project='elvo-198322')
    bucket = client.bucket('elvos')

    all_labels = load_labels(data_name, bucket)

    shuffled_blobs = list(
        bucket.list_blobs(prefix=f'processed3d/{data_name}/arrays/'))
    random.shuffle(shuffled_blobs)

    split_idx = int(0.8 * len(shuffled_blobs))
    training_blobs = shuffled_blobs[0:split_idx]
    validation_blobs = shuffled_blobs[split_idx:]

    x_train, y_train = load_split_data(training_blobs, all_labels)
    x_train_blob = bucket.blob(f'processed3d/{data_name}/x_train.npy')
    upload_arr(x_train, x_train_blob)
    y_train_blob = bucket.blob(f'processed3d/{data_name}/y_train.npy')
    upload_arr(y_train, y_train_blob)

    x_valid, y_valid = load_split_data(validation_blobs, all_labels)
    x_valid_blob = bucket.blob(f'processed3d/{data_name}/x_valid.npy')
    upload_arr(x_valid, x_valid_blob)
    y_valid_blob = bucket.blob(f'processed3d/{data_name}/y_valid.npy')
    upload_arr(y_valid, y_valid_blob)


def load_split_data(blobs, all_labels) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the blobs as a data array and a labels array

    :param blobs: an iterable of blobs containing numpy arrays
    :param all_labels: a nested dictionary with patient_id, then position
        strings (e.g. 32-64-160) as keys
    :return: a 4-D array with shape (2 * num positives, 32, 32, 32) and a
        1-D array of 1s and 0s
    """
    x_list = []
    y_list = []
    for blob in blobs:
        logging.info('downloading blob {}'.format(blob.name))
        with io.BytesIO() as stream:
            filename = blob.name.split('/')[-1]
            patient_id = filename[:-len('.csv')]

            blob.download_to_file(stream)
            stream.seek(0)
            arrays = np.load(stream)

            # TODO(luke): What if there's more than one occlusion?
            pair = positive_negative_pair(arrays, all_labels[patient_id])

            if pair is not None:
                x_list.extend(pair)
                y_list.extend([1, 0])

    return np.array(x_list), np.array(y_list)


def load_labels(data_name, bucket) -> Dict[str, Dict[str, bool]]:
    """
    Loads the labels from gs://<bucket>/processed3d/<data_name>/labels

    :param data_name:
    :param bucket:
    :return:
    """
    all_labels = {}
    for blob in bucket.list_blobs(prefix=f'processed3d/{data_name}/labels/'):
        filename = blob.name.split('/')[-1]
        patient_id = filename[:-len('.csv')]
        logging.info('downloading blob {}'.format(blob.name))

        tmp_filepath = '/tmp/{filename}'.format(filename=filename)
        blob.download_to_filename(tmp_filepath)
        patient_df = pd.read_csv(tmp_filepath)

        patient_labels = {}
        for _, row in patient_df.iterrows():
            patient_labels[row['position']] = row['label']
        all_labels[patient_id] = patient_labels

        os.remove(tmp_filepath)
    return all_labels


def upload_arr(arr: np.ndarray, blob: storage.Blob):
    logging.info(f'uploading blob {blob.name}')
    with io.BytesIO() as stream:
        # noinspection PyTypeChecker
        np.save(stream, arr)
        stream.seek(0)
        blob.upload_from_file(stream)


def positive_negative_pair(
        arrays, labels) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a pair of arrays from the list of arrays. One
    will be positive. The other is a negative.

    :param arrays: a dictionary of arrays with the position as the key
    :param labels: a dictionary of labels with the position as the key
    :return: the positive array and the negative array
    """
    for position, arr in arrays.items():
        if labels[position] == 1:
            logging.info('adding positive at position {}'.format(position))

            while True:
                parsed_pos = [int(p) for p in position.split('-')]
                logging.debug('parsed_pos: {}'.format(parsed_pos))
                # We choose one of the 6 chunks next to the positive
                offset = random.choice([-1, 1]) * 32
                index = random.randint(0, 2)
                parsed_pos[index] += offset
                nearby_pos_str = '-'.join([str(p) for p in parsed_pos])

                if nearby_pos_str not in labels:
                    logging.info('position {} is outside of the chunk, '
                                 'retrying'.format(nearby_pos_str))
                elif labels[nearby_pos_str] == 1:
                    logging.info('position {} is also positive, '
                                 'retrying'.format(nearby_pos_str))
                else:
                    # Found a negative, breaking
                    break

            logging.info(
                'adding negative at position {}'.format(nearby_pos_str))
            nearby_arr = arrays[nearby_pos_str]

            return arr, nearby_arr

    return None


default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 24),
}

dag = DAG(dag_id='preprocess3d_web',
          description='Preprocesses data using a configuration passed by'
                      ' the web app.',
          default_args=default_args,
          schedule_interval=None,
          catchup=False)

DATA_NAME = 'airflow-2'

preprocess_op = PythonOperator(
    task_id='create_chunks',
    python_callable=lambda: preprocess3d(DATA_NAME),
    dag=dag
)

split_op = PythonOperator(
    task_id='save_as_split_dirs',
    python_callable=lambda: save_as_split_dirs(DATA_NAME),
    dag=dag,
)

preprocess_op >> split_op

# TODO(luke): https://great-expectations.readthedocs.io/en/latest/glossary.html
# TODO: Option to skip certain operators based on last_modified
# TODO: Re-run dag to fix offset issue
# TODO: Upsampling, downsampling variations
