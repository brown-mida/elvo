import numpy as np
import logging
from matplotlib import pyplot as plt
import cloud_management as cloud
import transforms

location = 'numpy/axial'
prefix = location + '/'

def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

def set_cloud():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')
    return bucket


def get_og_mip(cropped_arr: np.ndarray):
    # perform the normal MIPing procedure
    not_extreme_arr = transforms.remove_extremes(cropped_arr)
    logging.info(f'removed array extremes')
    mip_arr = transforms.mip_normal(not_extreme_arr)

    plt.figure(figsize=(6, 6))
    plt.imshow(mip_arr, interpolation='none')
    plt.show()

def get_stripped_mip():
    for in_blob in set_cloud().list_blobs(prefix=prefix):

        # test: only using one patient's mipped scans for now
        # if in_blob.name != prefix + '0RB9KGMO90G1YQZD.npy':
        #     continue

        # perform the normal MIPing procedure
        logging.info(f'downloading {in_blob.name}')
        input_arr = cloud.download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")
        cropped_arr = transforms.crop_strip_skull(input_arr, location)
        logging.info(f"mipping numpy array")
        mip_arr = transforms.mip_normal(cropped_arr)

        print(mip_arr)

        # strip skull and segment blood vessels
        logging.info(f"segment blood vessels")
        stripped_arr = transforms.segment_vessels(mip_arr, location)

        print(stripped_arr)

        get_og_mip(cropped_arr)

        plt.figure(figsize=(6, 6))
        plt.imshow(stripped_arr, interpolation='none')
        plt.show()

get_stripped_mip()