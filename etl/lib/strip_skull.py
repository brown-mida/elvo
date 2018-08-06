"""
Script for segmenting and removing all irrelevant material from array version
of scans. In this context, "irrelevant material" is bone and brain (anything
other than vasculature).
"""

import numpy as np
import logging
from matplotlib import pyplot as plt
import cloud_management as cloud
import transforms

LOCATION = 'numpy/axial'
PREFIX = LOCATION + '/'


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_og_mip(cropped_arr: np.ndarray):
    """
    Gets and shows normal MIP for comparison

    :param cropped_arr: array to MIP
    :return:
    """
    # perform the normal MIPing procedure
    not_extreme_arr = transforms.remove_extremes(cropped_arr)
    logging.info("removed array extremes")
    mip_arr = transforms.mip_normal(not_extreme_arr)

    plt.figure(figsize=(6, 6))
    plt.imshow(mip_arr, interpolation='none')
    plt.show()


def save_to_cloud(arr: np.ndarray, in_blob):
    """
    Saves array to cloud

    :param arr: array to save
    :param in_blob: blob array comes from (for naming reasons)
    :return:
    """
    file_id = in_blob.name.split('/')[2]
    file_id = file_id.split('.')[0]
    print(file_id)
    cloud.save_stripped_npy(arr, file_id, "axial_single_channel")


def get_stripped_mip():
    """
    A script to get stripped and segmented MIP images to then do analysis on

    :return:
    """
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # for every blob
    for in_blob in bucket.list_blobs(prefix=PREFIX):

        # perform the normal MIPing procedure
        logging.info(f"downloading {in_blob.name}")
        input_arr = cloud.download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")
        cropped_arr = transforms.crop_strip_skull(input_arr, LOCATION)
        logging.info("mipping numpy array")
        mip_arr = transforms.mip_normal(cropped_arr)

        # strip skull and grey matter to segment blood vessels
        logging.info("segment blood vessels")
        stripped_arr = transforms.segment_vessels(mip_arr, LOCATION)

        # save to cloud
        save_to_cloud(stripped_arr, in_blob)

        # OPTIONAL: visualize stripped MIP
        # plt.figure(figsize=(6, 6))
        # plt.imshow(stripped_arr, interpolation='none')
        # plt.show()


if __name__ == '__main__':
    get_stripped_mip()
