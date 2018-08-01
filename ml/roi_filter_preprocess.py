import logging
import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from models.three_d import c3d
from google.cloud import storage
from tensorflow.python.lib.io import file_io


def download_array(blob: storage.Blob) -> np.ndarray:
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


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


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def create_chunks(annotations_df: pd.DataFrame, model):
    client = authenticate()
    bucket = client.get_bucket('elvos')

    # loop through every array on GCS
    for in_blob in bucket.list_blobs(prefix='airflow/npy'):
        # blacklist
        if in_blob.name == 'airflow/npy/LAUIHISOEZIM5ILF.npy':
            continue

        # get the file id
        file_id = in_blob.name.split('/')[2]
        file_id = file_id.split('.')[0]

        print(f'chunking {file_id}')
        # copy ROI if there's a positive match in the ROI annotations
        roi_df = annotations_df[
            annotations_df['patient_id'].str.match(file_id)]
        # if it's empty, this brain is ELVO negative
        if roi_df.empty:
            elvo_positive = False
        else:
            elvo_positive = True

        arr = download_array(in_blob)
        rois = []
        centers = []

        # if it's elvo positive
        if elvo_positive:
            for row in roi_df.itertuples():
                logging.info(row)
                """
                row[0] = index
                row[1] = patient ID
                row[2] = red1
                row[3] = red2
                row[4] = green1
                row[5] = green2
                row[6] = blue1
                row[7] = blue2
                """
                rois.append((int(len(arr) - row[7]),
                             int(row[4]),
                             int(row[2])))
                centers.append((int(((len(arr) - row[6])
                                     + (len(arr) - row[7])) / 2),
                                int((row[4] + row[5]) / 2),
                                int((row[2] + row[3]) / 2)))
            logging.info(rois, centers)

        h = 0
        # loop through every chunk
        for i in range(0, len(arr), 32):
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    found_positive = False

                    # loop through the available ROIs and centers
                    for roi, center in zip(rois, centers):

                        # if the center lies within this chunk
                        if i <= center[0] <= i + 32 \
                                and j <= center[1] <= j + 32 \
                                and k <= center[2] <= k + 32:
                            # save the ROI and skip this block
                            chunk = arr[roi[0]: roi[0] + 32,
                                        roi[1]: roi[1] + 32,
                                        roi[2]: roi[2] + 32]
                            save_chunks_to_cloud(np.asarray(chunk),
                                                 'filtered', 'positive',
                                                 file_id + str(h))
                            h += 1
                            found_positive = True

                    if found_positive:
                        continue

                    # copy the chunk
                    chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]

                    if np.asarray(chunk).shape != (32, 32, 32):
                        print(np.asarray(chunk).shape)
                        continue
                    # calculate the airspace
                    airspace = np.where(chunk < -300)
                    # if it's less than 90% airspace
                    if (airspace[0].size / chunk.size) < 0.9:
                        pred_chunk = np.expand_dims(chunk, axis=-1)
                        pred_chunk = np.expand_dims(pred_chunk, axis=0)
                        if model.predict(pred_chunk) > 0.4:
                            # save the label as 0 and save it to the cloud
                            save_chunks_to_cloud(np.asarray(chunk),
                                                 'filtered', 'negative',
                                                 file_id + str(h))

                    h += 1


def create_labels(annotations_df: pd.DataFrame, model):
    client = authenticate()
    bucket = client.get_bucket('elvos')
    label_dict = {}

    # loop through every array on GCS
    for in_blob in bucket.list_blobs(prefix='airflow/npy'):
        # blacklist
        if in_blob.name == 'airflow/npy/LAUIHISOEZIM5ILF.npy':
            continue

        # get the file id
        file_id = in_blob.name.split('/')[2]
        file_id = file_id.split('.')[0]

        logging.info(f'labeling {file_id}')

        # copy ROI if there's a positive match in the ROI annotations
        roi_df = annotations_df[
            annotations_df['patient_id'].str.match(file_id)]
        # if it's empty, this brain is ELVO negative
        if roi_df.empty:
            elvo_positive = False
        else:
            elvo_positive = True

        arr = download_array(in_blob)
        rois = []
        centers = []

        # if it's elvo positive
        if elvo_positive:

            for row in roi_df.itertuples():
                """
                row[0] = index
                row[1] = patient ID
                row[2] = red1
                row[3] = red2
                row[4] = green1
                row[5] = green2
                row[6] = blue1
                row[7] = blue2
                """
                rois.append((int(len(arr) - row[7]),
                             int(row[4]),
                             int(row[2])))
                centers.append((int(((len(arr) - row[6])
                                     + (len(arr) - row[7])) / 2),
                                int((row[4] + row[5]) / 2),
                                int((row[2] + row[3]) / 2)))

        # else it's elvo negative
        h = 0
        # loop through every chunk
        for i in range(0, len(arr), 32):
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    found_positive = False

                    # loop through the available ROIs and centers
                    for roi, center in zip(rois, centers):

                        # if the center lies within this chunk
                        if i <= center[0] <= i + 32 \
                                and j <= center[1] <= j + 32 \
                                and k <= center[2] <= k + 32:
                            # save the ROI and skip this block
                            label_dict[file_id + str(h)] = 1
                            h += 1
                            found_positive = True

                    if found_positive:
                        continue

                    # copy the chunk
                    chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                    if np.asarray(chunk).shape != (32, 32, 32):
                        print(np.asarray(chunk).shape)
                        continue
                    # calculate the airspace
                    airspace = np.where(chunk < -300)
                    # if it's less than 90% airspace
                    if (airspace[0].size / chunk.size) < 0.9:
                        pred_chunk = np.expand_dims(chunk, axis=-1)
                        pred_chunk = np.expand_dims(pred_chunk, axis=0)
                        if model.predict(pred_chunk) > 0.4:
                            # save the label as 0 and save it to the cloud
                            label_dict[file_id + str(h)] = 0
                    h += 1

    # convert the labels to a df
    labels_df = pd.DataFrame.from_dict(label_dict, orient='index',
                                       columns=['label'])
    labels_df.to_csv('annotated_labels.csv')


def process_labels():
    annotations_df = pd.read_csv(
        '/home/harold_triedman/elvo-analysis/annotations.csv')
    # annotations_df = pd.read_csv(
    #         '/Users/haltriedman/Desktop/annotations.csv')
    annotations_df = annotations_df.drop(['created_by',
                                          'created_at',
                                          'ROI Link',
                                          'Unnamed: 10'],
                                         axis=1)
    annotations_df = annotations_df[
        annotations_df.red1 == annotations_df.red1]
    logging.info(annotations_df)
    return annotations_df


def inspect_rois(annotations_df):
    client = authenticate()
    bucket = client.get_bucket('elvos')

    # loop through every array on GCS
    for in_blob in bucket.list_blobs(prefix='airflow/npy'):
        # if in_blob.name != 'airflow/npy/ZZX0ZNWG6Q9I18GK.npy':
        #     continue
        # blacklist
        if in_blob.name == 'airflow/npy/LAUIHISOEZIM5ILF.npy':
            continue

        # get the file id
        file_id = in_blob.name.split('/')[2]
        file_id = file_id.split('.')[0]

        logging.info(f'chunking {file_id}')
        # copy ROI if there's a positive match in the ROI annotations
        roi_df = annotations_df[
            annotations_df['patient_id'].str.match(file_id)]
        # if it's empty, this brain is ELVO negative
        if roi_df.empty:
            elvo_positive = False
        else:
            elvo_positive = True

        arr = download_array(in_blob)

        # if it's elvo positive
        if elvo_positive:
            chunks = []
            blue = int(len(arr) - roi_df['blue2'].iloc[0])
            green = int(roi_df['green1'].iloc[0])
            red = int(roi_df['red1'].iloc[0])
            chunks.append(arr[blue: blue + 32,
                          green: green + 50, red: red + 50])
            chunks.append(arr[
                          blue: blue + 32, red: red + 50, green: green + 50])
            start = 0
            for chunk in chunks:
                logging.info(start)
                axial = np.max(chunk, axis=0)
                coronal = np.max(chunk, axis=1)
                sag = np.max(chunk, axis=2)
                fig, ax = plt.subplots(1, 3, figsize=(6, 4))
                ax[0].imshow(axial, interpolation='none')
                ax[1].imshow(coronal, interpolation='none')
                ax[2].imshow(sag, interpolation='none')
                plt.show()
                start += 10


def run_preprocess():
    configure_logger()
    model = c3d.C3DBuilder.build()
    model.load_weights('tmp/FINAL_RUN_6.hdf5')
    annotations_df = process_labels()
    create_labels(annotations_df, model)
    create_chunks(annotations_df, model)
    # inspect_rois(annotations_df)


if __name__ == '__main__':
    run_preprocess()
