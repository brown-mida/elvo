"""
This script visualizes predictions by slice based off of the high performing
model in the first run through of training on non-pared-down data
"""
import io
from matplotlib import pyplot as plt
import math
from models.three_d import c3d
import tensorflow as tf
from google.cloud import storage
import numpy as np


def download_array(blob: storage.Blob) -> np.ndarray:
    """Downloads data blobs as numpy arrays

    :param blob: the GCS blob you want to download as an array
    :return:
    """
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def display_preds(preds):
    """
    Actually displays the predictions from make_preds()
    :param preds: a list of 3D predictions by scan
    :return:
    """
    # Get npy files from Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        # Use this when running on VM
        # '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        # Use this when running locally
        'credentials/client_secret.json'
    )
    bucket = gcs_client.get_bucket('elvos')
    blobs = bucket.list_blobs(prefix='airflow/test_npy/')

    # Get every scan in airflow/test_npy
    for h, blob in enumerate(blobs):
        if blob.name != 'airflow/test_npy/GJ35FZQ5DSP09A4L.npy':
            continue
        if not blob.name.endswith('.npy'):
            continue
        arr = download_array(blob)
        pred = preds[0]

        # Go through each 32-voxel thick slice
        for a, i in enumerate(range(0, len(arr), 32)):
            # Make a mip of the slice
            if i + 32 > len(arr):
                mip = np.asarray(arr[i:])
            else:
                mip = np.asarray(arr[i:i+32])
            mip = np.max(mip, axis=0)
            # Expand the predictions to a heatmap by a factor of 32x32
            hm = np.kron(pred[a], np.ones(shape=(32, 32)))
            # Plot each slice
            fig = plt.figure(figsize=(12, 7))
            fig.add_subplot(1, 2, 1)
            plt.imshow(mip, interpolation='none')
            fig.add_subplot(1, 2, 2)
            plt.imshow(hm)
            fig.tight_layout()
            plt.plot()
            plt.show()


def make_preds():
    """
    Loops through every single array in airflow/test_npy/ and makes predictions
    about each chunk in them using the highest performing first run-through
    model
    :return: A list of 3D predictions for each scan
    """
    # Get npy files from Google Cloud Storage
    gcs_client = storage.Client.from_service_account_json(
        # Use this when running on VM
        # '/home/harold_triedman/elvo-analysis/credentials/client_secret.json'

        # Use this when running locally
        'credentials/client_secret.json'
    )
    bucket = gcs_client.get_bucket('elvos')

    # load model
    model = c3d.C3DBuilder.build()
    # model.load_weights('tmp/FINAL_RUN_6.hdf5')
    model.load_weights('/Users/haltriedman/Desktop/FINAL_RUN_6.hdf5')
    metapreds = []
    blobs = bucket.list_blobs(prefix='airflow/test_npy/')

    # Get every scan in airflow/npy
    for h, blob in enumerate(blobs):
        print(blob.name)
        if blob.name != 'airflow/test_npy/GJ35FZQ5DSP09A4L.npy':
            continue
        print('hi')
        if not blob.name.endswith('.npy'):
            continue
        arr = download_array(blob)

        # Get every chunk in the scan
        preds = []
        for i in range(0, len(arr), 32):
            # get a list of 0s as long as the number of chunks in this slice
            layer = np.zeros(shape=(int(math.ceil(len(arr[0]) / 32) *
                                        math.ceil(len(arr[0][0]) / 32))))
            layer_idx = 0
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    chunk = arr[i: i + 32,
                                j: j + 32,
                                k: k + 32]
                    airspace = np.where(chunk < -300)
                    # if it's less than 90% airspace
                    if (airspace[0].size / chunk.size) < 0.9:
                        # make a prediction about it using the model
                        if chunk.shape == (32, 32, 32):
                            chunk = np.expand_dims(chunk, axis=-1)
                            chunk = np.expand_dims(chunk, axis=0)
                            layer[layer_idx] = model.predict(chunk)
                    layer_idx += 1
            # reshape the list into a slice and append it to this scan's preds
            layer = layer.reshape((int(math.ceil(len(arr[0]) / 32)),
                                   int(math.ceil(len(arr[0][0]) / 32))))
            preds.append(layer)

        preds = np.asarray(preds)
        # append this scan's preds to metapreds
        metapreds.append(preds)
    return np.asarray(metapreds)


def main():
    tf.Session(config=tf.ConfigProto(log_device_placement=True))
    preds = make_preds()
    display_preds(preds)


if __name__ == '__main__':
    main()
