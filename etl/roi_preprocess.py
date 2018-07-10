from lib import roi_transforms, transforms, cloud_management as cloud
import logging
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt


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

    # loop through every array on GCS
    for in_blob in bucket.list_blobs(prefix='airflow/npy'):
        # blacklist
        if in_blob.name == 'airflow/npy/LAUIHISOEZIM5ILF.npy':
            continue

        # get the file id
        file_id = in_blob.name.split('/')[2]
        file_id = file_id.split('.')[0]
        # copy ROI if there's a positive match in the ROI annotations
        roi_df = annotations_df[annotations_df['patient_id'].str.match(file_id)]
        # if it's empty, this brain is ELVO negative
        if roi_df.empty:
            elvo_positive = False
        else:
            elvo_positive = True

        # do preprocessing
        arr = cloud.download_array(in_blob)
        stripped = transforms.segment_vessels(arr)
        point_cloud = transforms.point_cloud(arr)

        # if it's elvo positive
        if elvo_positive:
            h = 0
            # loop through every chunk
            for i in range(int(roi_df['blue1'].iloc[0]) % 32, len(arr), 32):
                for j in range(int(roi_df['green1'].iloc[0]) % 32, len(arr[0]), 32):
                    for k in range(int(roi_df['red1'].iloc[0]) % 32, len(arr[0][0]), 32):

                        elvo_chunk = False

                        # if the chunk contains the elvo, set a boolean to true
                        if i == int(roi_df['blue1'].iloc[0]) \
                                and j == int(roi_df['green1'].iloc[0]) \
                                and k == int(roi_df['red1'].iloc[0]):
                            print('hi')
                            elvo_chunk = True

                        # copy the chunk
                        chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                        # calculate airspace
                        airspace = np.where(chunk < -300)
                        # if the chunk is more than 90% airspace
                        if (airspace[0].size / chunk.size) < 0.9:
                            # if it's the positive chunk, set the label to
                            #   1 and save
                            if elvo_chunk:
                                cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                           'normal/positive',
                                                           file_id + str(h))

                            # else set the label to 0 and save
                            else:
                                cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                           'normal/negative',
                                                           file_id + str(h))

                        # do the same thing with stripped array
                        stripped_chunk = stripped[i:(i + 32), j:(j + 32), k:(k + 32)]
                        stripped_airspace = np.where(stripped_chunk <= -50)
                        if (stripped_airspace[0].size / stripped_chunk.size) < 0.9:
                            if elvo_chunk:
                                cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                           'stripped/positive',
                                                           file_id + str(h))
                            else:
                                cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                           'stripped/negative',
                                                           file_id + str(h))

                        # do the same thing with point cloud array
                        pc_chunk = point_cloud[i:(i + 32), j:(j + 32), k:(k + 32)]
                        pc_airspace = np.where(pc_chunk == 0)
                        if (pc_airspace[0].size / pc_chunk.size) < 0.9:
                            if elvo_chunk:
                                cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                           'point_cloud/positive',
                                                           file_id + str(h))
                            else:
                                cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                           'point_cloud/negative',
                                                           file_id + str(h))
                        h += 1

        # else it's elvo negative
        else:
            h = 0
            # loop through every chunk
            for i in range(0, len(arr), 32):
                for j in range(0, len(arr[0]), 32):
                    for k in range(0, len(arr[0][0]), 32):

                        # copy the chunk
                        chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                        # calculate the airspace
                        airspace = np.where(chunk < -300)
                        # if it's less than 90% airspace
                        if (airspace[0].size / chunk.size) < 0.9:
                            # save the label as 0 and save it to the cloud
                            cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                       'normal/negative',
                                                       file_id + str(h))

                        # do the same thing for stripped
                        stripped_chunk = \
                            stripped[i:(i + 32), j:(j + 32), k:(k + 32)]
                        stripped_airspace = np.where(stripped_chunk <= -50)
                        if (stripped_airspace[0].size / stripped_chunk.size) \
                                < 0.9:
                            cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                       'stripped/negative',
                                                       file_id + str(h))

                        # do the same thing for point cloud
                        pc_chunk = \
                            point_cloud[i:(i + 32), j:(j + 32), k:(k + 32)]
                        pc_airspace = np.where(pc_chunk == 0)
                        if (pc_airspace[0].size / pc_chunk.size) < 0.9:
                            cloud.save_chunks_to_cloud(np.asarray(chunk),
                                                       'point_cloud/negative',
                                                       file_id + str(h))
                        h += 1


def create_labels(annotations_df: pd.DataFrame):
    client = cloud.authenticate()
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
        # copy ROI if there's a positive match in the ROI annotations
        roi_df = annotations_df[annotations_df['patient_id'].str.match(file_id)]
        # if it's empty, this brain is ELVO negative
        if roi_df.empty:
            elvo_positive = False
        else:
            elvo_positive = True

        # do preprocessing
        arr = cloud.download_array(in_blob)

        # if it's elvo positive
        if elvo_positive:
            h = 0
            # loop through every chunk
            for i in range(int(roi_df['blue1'].iloc[0]) % 32, len(arr), 32):
                for j in range(int(roi_df['green1'].iloc[0]) % 32, len(arr[0]), 32):
                    for k in range(int(roi_df['red1'].iloc[0]) % 32, len(arr[0][0]), 32):

                        elvo_chunk = False

                        # if the chunk contains the elvo, set a boolean to true
                        if i == int(roi_df['blue1'].iloc[0]) \
                                and j == int(roi_df['green1'].iloc[0]) \
                                and k == int(roi_df['red1'].iloc[0]):
                            elvo_chunk = True

                        # copy the chunk
                        chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                        # calculate airspace
                        airspace = np.where(chunk < -300)
                        # if the chunk is more than 90% airspace
                        if (airspace[0].size / chunk.size) < 0.9:
                            # if it's the positive chunk, set the label to 1
                            if elvo_chunk:
                                label_dict[file_id + str(h)] = 1

                            # else set the label to 0
                            else:
                                label_dict[file_id + str(h)] = 0
                        h += 1

        # else it's elvo negative
        else:
            h = 0
            # loop through every chunk
            for i in range(0, len(arr), 32):
                for j in range(0, len(arr[0]), 32):
                    for k in range(0, len(arr[0][0]), 32):
                        # copy the chunk
                        chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                        # calculate the airspace
                        airspace = np.where(chunk < -300)
                        # if it's less than 90% airspace
                        if (airspace[0].size / chunk.size) < 0.9:
                            # set the label as 0
                            label_dict[file_id + str(h)] = 0

                        h += 1

    # convert the labels to a df
    labels_df = pd.DataFrame.from_dict(label_dict, orient='index', columns=['label'])
    print(labels_df)
    labels_df.to_csv('annotated_labels.csv')


def pd_test():
    label_dict = {'04IOS24JP70LHBGB182': 0,
                  '04IOS24JP70LHBGB183': 0,
                  '04IOS24JP70LHBGB184': 1,
                  '04IOS24JP70LHBGB189': 0}
    print(pd.DataFrame.from_dict(label_dict, orient='index', columns=['label']))


def process_labels():
    annotations_df = pd.read_csv('/home/harold_triedman/elvo-analysis/annotations.csv')
    annotations_df = annotations_df.drop(['created_by',
                                          'created_at',
                                          'ROI Link',
                                          'Unnamed: 10',
                                          'Mark here if Matt should review'],
                                         axis=1)
    annotations_df = annotations_df[annotations_df.red1 == annotations_df.red1]
    print(annotations_df)
    return annotations_df


if __name__ == '__main__':
    configure_logger()
    annotations_df = process_labels()
    create_labels(annotations_df)
    create_chunks(annotations_df)
