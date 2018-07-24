import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib import cloud_management as cloud  # , roi_transforms, transforms


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def process_labels():
    annotations_df = pd.read_csv(
        '/home/amy/annotations.csv')
    # annotations_df = pd.read_csv(
    #         '/Users/haltriedman/Desktop/annotations.csv')
    annotations_df = annotations_df.drop(['created_by',
                                          'created_at',
                                          'ROI Link',
                                          'Unnamed: 10',
                                          'Mark here if Matt should review'],
                                         axis=1)
    annotations_df = annotations_df[
        annotations_df.red1 == annotations_df.red1]
    logging.info(annotations_df)
    return annotations_df


def inspect_rois(annotations_df):
    client = cloud.authenticate()
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

        arr = cloud.download_array(in_blob)

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
    annotations_df = process_labels()
    create_labels(annotations_df)


if __name__ == '__main__':
    run_preprocess()
