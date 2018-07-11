"""
Purpose: This script takes all the positive ROI-cropped 32x32x32 "chunks,"
transforms them via vertical flips and rotations, and adds them to the GCS
storage folder "positives + augmentation."
"""

import logging
from matplotlib import pyplot as plt
from lib import transforms, cloud_management as cloud
import random
import numpy as np
import pandas as pd
import itertools

def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

def transform_one(arr, file_id):
    iterlist = list(itertools.product('01', repeat = 3))
    axes = [[0, 1], [1, 2], [0, 2]]

    transform_number = 0
    for i in axes:
        rotated = np.rot90(arr, axes=i)
        for j in iterlist:
            transform_number += 1
            if j[0] == '1':
                flipped = np.flipud(rotated)
            if j[1] == '1':
                flipped = np.fliplr(rotated)
            if j[2] == '1':
                flipped = rotated[:, :, ::-1]
                # save to the numpy generator source directory
            file_id_new =  file_id + "_" + str(transform_number)
            print(file_id_new)
            #cloud.save_chunks_to_cloud(flipped,
                                      # 'normal', 'positive', file_id_new)

def transform_positives():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # iterate through every source directory...
    prefix = "chunk_data/normal/positive"
    logging.info(f"transforming positive chunks from {prefix}")

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
        flipped = transform_one(input_arr, file_id)

        # plt.figure(figsize=(6, 6))
        # plt.imshow(not_extreme_arr, interpolation='none')
        # plt.show()

if __name__ == '__main__':
    configure_logger()
    transform_positives()