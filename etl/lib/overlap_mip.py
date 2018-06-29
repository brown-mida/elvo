"""
Purpose: This script implements maximum intensity projections (MIP). This
process involves taking 3D brain scans, chunking them into three relevant
sections, and compressing each section's maximum values down into a 2D array.
When these are recombined, we get an array with the shape(3, X, Y) — which is
ready to be fed directly into standardized Keras architecture with pretrained
feature detection weights from ImageNet/CIFAR10.
"""

import logging
from matplotlib import pyplot as plt
import cloud_management as cloud
import transforms

WHENCE = ['numpy/axial',
          'numpy/coronal']


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


if __name__ == '__main__':
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # iterate through every source directory...
    for location in WHENCE:
        prefix = location + '/'
        logging.info(f"MIPing images from {prefix}")

        for in_blob in bucket.list_blobs(prefix=prefix):
            # blacklist
            if in_blob.name == prefix + 'LAUIHISOEZIM5ILF.npy':
                continue

            # perform the normal MIPing procedure
            logging.info(f'downloading {in_blob.name}')
            input_arr = cloud.download_array(in_blob)
            logging.info(f"blob shape: {input_arr.shape}")
            if location == 'numpy/axial':
                cropped_arr = transforms.crop_overlap_axial(input_arr,
                                                            location)
            else:
                cropped_arr = transforms.crop_overlap_coronal(input_arr,
                                                              location)
            not_extreme_arr = transforms.remove_extremes(cropped_arr)
            logging.info(f'removed array extremes')
            mip_arr = transforms.mip_overlap(not_extreme_arr)
            plt.figure(figsize=(6, 6))
            plt.imshow(mip_arr[10], interpolation='none')
            plt.show()

            # if the source directory is one of the luke ones
            # if location != 'numpy':
            #     file_id = in_blob.name.split('/')[2]
            #     file_id = file_id.split('.')[0]
            #     # save to both a training and validation split
            #     # and a potential generator source directory
            #     cloud.save_npy_to_cloud(mip_arr, file_id, 'processed')
            # # otherwise it's from numpy
            # else:
            file_id = in_blob.name.split('/')[2]
            file_id = file_id.split('.')[0]
            # save to the numpy generator source directory
            cloud.save_npy_to_cloud(mip_arr, file_id, location, 'overlap')
