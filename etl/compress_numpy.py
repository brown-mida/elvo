import io
import logging
from typing import Dict

import numpy as np
from google.cloud import storage


def save_arrays(arrays: Dict[str, np.ndarray],
                filename: str,
                bucket: storage.Bucket):
    out_stream = io.BytesIO()
    np.savez_compressed(out_stream, **arrays)
    out_stream.seek(0)
    out_blob = bucket.blob(filename)
    out_blob.upload_from_file(out_stream)


def compress_numpy(in_dir, out_dir):
    """Saves the arrays in batches of 10.
    """
    client = storage.Client(project='elvo-198322')
    bucket = client.get_bucket('elvos')

    blob: storage.Blob
    arrays = {}
    i = 0
    for blob in bucket.list_blobs(prefix=in_dir):
        patient_id = blob.name[len(in_dir): -len('.npy')]
        if len(arrays) >= 10:
            logging.info(f'uploading arrays: {list(arrays.keys())}')
            save_arrays(arrays,
                        f'{out_dir}{i}.npz',
                        bucket)
            arrays = {}
            i += 1
        in_stream = io.BytesIO()
        logging.info(f'downloading {blob.name}, patient id: {patient_id}')
        blob.download_to_file(in_stream)
        in_stream.seek(0)
        arr = np.load(in_stream)
        arrays[patient_id] = arr

    if len(arrays) > 0:  # Upload remaining files
        logging.info(f'uploading arrays: {list(arrays.keys())}')
        save_arrays(arrays,
                    f'{out_dir}{i}.npz',
                    bucket)
