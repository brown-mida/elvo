import io
from matplotlib import pyplot as plt
import math
from models.three_d import c3d
import tensorflow as tf
from google.cloud import storage
import numpy as np
from image_slice_viewer import IndexTracker


def download_array(blob: storage.Blob) -> np.ndarray:
    """Downloads data blobs as numpy arrays

    :param blob: the GCS blob you want to download as an array
    :return:
    """
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def display_preds(probs, probs_90):
    """
    Actually displays the predictions from make_preds()
    :param probs: a list of 3D predictions by scan
    :param probs_90: a masked list of 3D predictions by scan, only showing
    those > 0.9
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
        if blob.name != 'airflow/test_npy/04IOS24JP70LHBGB.npy':
            continue

        arr = download_array(blob)
        pred = probs[0]
        pred_90 = probs_90[0]

        mips = []
        preds = []
        preds_90 = []
        # For each 32-voxel-thick slice
        for a, i in enumerate(range(0, len(arr), 32)):
            # Make a MIP
            if len(arr) - 32 < 0:
                break
            if i + 32 > len(arr):
                mip = np.asarray(arr[i:])
            else:
                mip = np.asarray(arr[i:i+32])
            mip = np.max(mip, axis=0)
            # Make a heatmap by scaling up pred by a factor of 32x32
            hm = np.kron(pred[a], np.ones(shape=(32, 32)))
            hm_90 = np.kron(pred_90[a], np.ones(shape=(32, 32)))

            # Make figure to plot within
            fig = plt.figure(figsize=(9, 7))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            mips.append(mip)
            preds.append(hm)
            preds_90.append(hm_90)

        # Transpose the array so that scroller works
        mips = np.transpose(np.asarray(mips), (1, 2, 0))
        preds = np.transpose(np.asarray(preds), (1, 2, 0))
        preds_90 = np.transpose(np.asarray(preds_90), (1, 2, 0))

        # Make an index tracker to allow people to scroll through the brain
        tracker = IndexTracker(ax1, ax2, mips, preds)
        fig.tight_layout()
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

        # TODO: get probs_90 display to work
        # print(preds_90)
        # tracker_90 = IndexTracker(ax1, ax2, mips, preds_90)
        # fig.tight_layout()
        # fig.canvas.mpl_connect('scroll_event', tracker_90.onscroll)
        # plt.show()


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
    model.load_weights('/Users/haltriedman/Desktop/hard_training_8.hdf5')
    metapreds = []
    metapreds_90 = []
    blobs = bucket.list_blobs(prefix='airflow/test_npy/')

    # Get every scan in airflow/npy
    for h, blob in enumerate(blobs):
        if blob.name != 'airflow/test_npy/04IOS24JP70LHBGB.npy':
            continue
        if not blob.name.endswith('.npy'):
            continue
        arr = download_array(blob)

        # Get every chunk in the scan
        preds = []
        preds_90 = []
        for i in range(0, len(arr), 32):
            # Make a layer of that slice size // 32
            layer = np.zeros(shape=(int(math.ceil(len(arr[0]) / 32)
                                        * math.ceil(len(arr[0][0]) / 32))))
            layer_90 = np.zeros(shape=(int(math.ceil(len(arr[0]) / 32)
                                           * math.ceil(len(arr[0][0]) / 32))))
            layer_idx = 0
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    chunk = arr[i: i + 32,
                                j: j + 32,
                                k: k + 32]
                    airspace = np.where(chunk < -300)
                    # if it's less than 90% airspace
                    if (airspace[0].size / chunk.size) < 0.9:
                        if chunk.shape == (32, 32, 32):
                            # make a prediction about it
                            chunk = np.expand_dims(chunk, axis=-1)
                            chunk = np.expand_dims(chunk, axis=0)
                            prob = model.predict(chunk)
                            layer[layer_idx] = prob
                            # (maybe) add it to layer_90
                            if prob > 0.9:
                                print(prob)
                                layer_90[layer_idx] = prob
                    layer_idx += 1

            # reshape to a slice
            layer = np.reshape(layer, (int(math.ceil(len(arr[0]) / 32)),
                                       int(math.ceil(len(arr[0][0]) / 32))))
            layer_90 = np.reshape(layer_90, (int(math.ceil(len(arr[0]) / 32)),
                                             int(math.ceil(
                                                 len(arr[0][0]) / 32))))
            # append the layer to the set of preds for the scan
            preds.append(layer)
            preds_90.append(layer_90)

        preds = np.asarray(preds)
        preds_90 = np.asarray(preds_90)
        # append the scan to metapreds
        metapreds.append(preds)
        metapreds_90.append(preds_90)
    return np.asarray(metapreds), np.asarray(metapreds_90)


def main():
    tf.Session(config=tf.ConfigProto(log_device_placement=True))
    probs, probs_90 = make_preds()
    display_preds(probs, probs_90)


if __name__ == '__main__':
    main()
