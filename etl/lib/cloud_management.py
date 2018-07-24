import logging
import io
from tensorflow.python.lib.io import file_io
# import imageio
import numpy as np
from google.cloud import storage


def authenticate():
    return storage.Client.from_service_account_json(
        # for running on airflow GPU
        # '/home/lukezhu/elvo-analysis/credentials/client_secret.json'
        
        # for running on hal's GPU
        '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        # for running on amy's GPU
        # '/home/amy/credentials/client_secret.json'

        # for running locally
        # 'credentials/client_secret.json'
    )


def download_array(blob: storage.Blob) -> np.ndarray:
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def upload_png(arr: np.ndarray, id: str, type: str, bucket: storage.Bucket):
    """Uploads MIP PNGs to gs://elvos/mip_data/<patient_id>/<scan_type>_mip.png.
    """
    try:
        out_stream = io.BytesIO()
        # imageio.imwrite(out_stream, arr, format='png')
        out_filename = f'mip_data/{id}/{type}_mip.png'
        print(out_filename)
        out_blob = storage.Blob(out_filename, bucket)
        out_stream.seek(0)
        out_blob.upload_from_file(out_stream)
        print("Saved png file.")
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')


def save_npy_to_cloud(arr: np.ndarray, id: str, type: str, view: str):
    """Uploads MIP .npy files to gs://elvos/mip_data/from_numpy/<patient
        id>_mip.npy
    """
    try:
        perspective = type.split('/')[1]
        print(f'gs://elvos/mip_data/{view}/{perspective}/{id}.npy')
        np.save(file_io.FileIO(f'gs://elvos/mip_data/{view}/{perspective}/'
                               f'{id}.npy',
                               'w'), arr)
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')


def save_segmented_npy_to_cloud(arr: np.ndarray,
                                id: str, type: str, view: str):
    """Uploads MIP .npy files to gs://elvos/mip_data/from_numpy/<patient
        id>_mip.npy
    """
    try:
        perspective = type.split('/')[1]
        print(f'gs://elvos/stripped_mip_data/{view}/{perspective}/{id}.npy')
        np.save(file_io.FileIO(
            f'gs://elvos/stripped_mip_data/{view}/{perspective}/'
            f'{id}.npy',
            'w'), arr)
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')


def save_roi_npy(arr: np.ndarray, id: str, type: str, view: str):
    """Uploads ROI-cropped .npy files to gs://elvos/roi_data/{view}
        /{perspective}/<patient
        id>_mip.npy
    """
    try:
        perspective = type.split('/')[1]
        print(f'gs://elvos/roi_data/{view}/{perspective}/{id}.npy')
        np.save(file_io.FileIO(f'gs://elvos/roi_data/{view}/{perspective}/'
                               f'{id}.npy',
                               'w'), arr)
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')


def save_chunks_to_cloud(arr: np.ndarray, type: str,
                         elvo_status: str, id: str):
    """Uploads chunk .npy files to gs://elvos/chunk_data/<patient_id>.npy
    """
    try:
        print(f'gs://elvos/chunk_data/{type}/{elvo_status}/{id}.npy')
        np.save(file_io.FileIO(f'gs://elvos/chunk_data/{type}/'
                               f'{elvo_status}/{id}.npy', 'w'), arr)
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')
