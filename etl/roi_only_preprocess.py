"""
Purpose: This script implements maximum intensity projections (MIP). This
process involves taking 3D brain scans and compressing their maximum values
down into a single 2D array.
"""

# TODO: preprocess coronal and sagittal scans so they have mips too
import logging
from matplotlib import pyplot as plt
from lib import transforms, cloud_management as cloud
import random

def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# identify cropping coordinates for a single image with
# the given ID
def roi_crop_one(arr: np.ndarray, file_id: str):
    # copy ROI if there's a positive match in the ROI annotations
    roi_df = annotations_df[annotations_df['patient_id'].str.match(file_id)]
    to_return = arr
    # if it's empty, this brain is ELVO negative
    if roi_df.empty:
        elvo_positive = False
    else:
        elvo_positive = True

    # if it's elvo positive
    if elvo_positive:
        h = 0
        # loop through every chunk
        x1 = int(roi_df['blue1'].iloc[0])
        y1 = int(roi_df['green1'].iloc[0])
        z1 = int(roi_df['red1'].iloc[0])
        to_return = arr[x1:(x1 + 32), y1:(y1 + 32), z1:(z1 + 32)]
    # else it's elvo negative
    else:
        shape = arr.shape
        x1 = random.randint((shape(0)/4), (shape(0) - (shape(0)/4)))
        y1 = random.randint((shape(1)/4), (shape(1) - (shape(1)/4)))
        z1 = random.randint((shape(2)/4), (shape(2) - (shape(2)/4)))
        to_return = arr[x1:(x1 + 32), y1:(y1 + 32), z1:(z1 + 32)]
    return toreturn

def roi_crop_all():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # iterate through every source directory...
    logging.info(f"cropping images from airflow/npy to ROI region")

        for in_blob in bucket.list_blobs(prefix=prefix):
            # blacklist
            if in_blob.name == prefix + 'LAUIHISOEZIM5ILF.npy':
                continue

            # perform the normal cropping procedure
            logging.info(f'downloading {in_blob.name}')
            file_id = in_blob.name.split('/')[2]
            file_id = file_id.split('.')[0]

            input_arr = cloud.download_array(in_blob)
            logging.info(f"blob shape: {input_arr.shape}")
            # crop individual input array
            cropped_arr = roi_crop_one(input_arr, file_id)

            not_extreme_arr = transforms.remove_extremes(cropped_arr)
            logging.info(f'removed array extremes')
            plt.figure(figsize=(6, 6))
            plt.imshow(not_extreme_arr, interpolation='none')
            plt.show()


            # save to the numpy generator source directory
            cloud.save_roi_npy(mip_arr, file_id, location, 'normal')

def process_labels():
    annotations_df = pd.read_csv('/home/amy/data/annotations.csv')
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
    roi_crop_all()
