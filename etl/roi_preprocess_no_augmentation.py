import logging

import numpy as np
import pandas as pd

from lib import cloud_management as cloud  # , roi_transforms, transforms


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def create_chunks(annotations_df: pd.DataFrame):
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # loop through every positive array on GCS
    # no need to loop through negatives, as those are fine in their
    # current state
    for in_blob in bucket.list_blobs(prefix='chunk_data/normal/positive'):

        # get the file id
        file_id = in_blob.name.split('/')[3]
        file_id = file_id.split('.')[0]

        logging.info(f'getting {file_id}')
        # copy region if it's the original image, not a rotation/
        # reflection

        if '_1' in file_id:
            print("HIII")
            arr = cloud.download_array(in_blob)
            logging.info(f'downloading {file_id}')
            cloud.save_chunks_to_cloud(arr, 'normal',
                                       'positive_no_aug',
                                        file_id)


def create_labels(annotations_df: pd.DataFrame):
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')
    label_dict = {}

    print("HELLO")
    # loop through every array on GCS
    for in_blob in bucket.list_blobs(prefix='/npy'):
        print("HELLO 2")
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

        arr = cloud.download_array(in_blob)
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
                            label_dict[file_id + str(h) + '_1'] = 1
                            h += 1
                            found_positive = True

                    if found_positive:
                        continue

                    # copy the chunk
                    chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                    # calculate the airspace
                    airspace = np.where(chunk < -300)
                    # if it's less than 90% airspace
                    if (airspace[0].size / chunk.size) < 0.9:
                        # save the label as 0 and save it to the cloud
                        label_dict[file_id + str(h)] = 0
                    h += 1

    # convert the labels to a df
    labels_df = pd.DataFrame.from_dict(label_dict, orient='index',
                                       columns=['label'])
    labels_df.to_csv('/home/amy/data/no_aug_annotated_labels.csv')


def process_labels():
    annotations_df = pd.read_csv(
        '/home/amy/elvo-analysis/annotations.csv')
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


def run_preprocess():
    configure_logger()
    annotations_df = process_labels()
    create_labels(annotations_df)
    create_chunks(annotations_df)
    # inspect_rois(annotations_df)


if __name__ == '__main__':
    run_preprocess()
