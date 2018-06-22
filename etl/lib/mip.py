"""
Purpose: This script implements maximum intensity projections (MIP). This
process involves taking 3D brain scans and compressing their maximum values
down into a single 2D array.
"""

# TODO: preprocess coronal and sagittal scans so they have mips too
import logging
import numpy as np
# from matplotlib import pyplot as plt
import etl.lib.cloud_management as cloud
import etl.lib.transforms as transforms


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

    # from numpy directory
    for in_blob in bucket.list_blobs(prefix='numpy/'):

        # blacklist
        if in_blob.name == 'numpy/LAUIHISOEZIM5ILF.npy':
            continue

        logging.info(f'downloading {in_blob.name}')
        input_arr = cloud.download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")

        cropped_arr = transforms.crop(input_arr)
        not_extreme_arr = transforms.remove_extremes(cropped_arr)

        logging.info(f'removed array extremes')
        # create folder w patient ID
        axial = transforms.mip_array(not_extreme_arr, 'numpy', 'axial')
        logging.info(f'mip-ed CTA image')
        normalized = transforms.normalize(axial, lower_bound=-400)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(axial, interpolation='none')
        # plt.show()
        file_id = in_blob.name.split('/')[1]
        file_id = file_id.split('.')[0]

        cloud.save_npy_to_cloud(axial, in_blob.name[6:22], 'numpy')
        logging.info(f'saved .npy file to cloud')

    # from preprocess_luke/training directory
    for in_blob in bucket.list_blobs(prefix='preprocess_luke/training/'):
        if in_blob.name == 'numpy/LAUIHISOEZIM5ILF.npy':
            continue
        logging.info(f'downloading {in_blob.name}')
        input_arr = cloud.download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")
        transposed_arr = np.transpose(input_arr, (2, 0, 1))
        logging.info(f'transposed to: {transposed_arr.shape}')

        cropped_arr = transforms.crop(transposed_arr, 'luke')
        not_extreme_arr = transforms.remove_extremes(cropped_arr)

        logging.info(f'removed array extremes')
        # create folder w patient ID
        axial = transforms.mip_array(not_extreme_arr, 'axial')
        logging.info(f'mip-ed CTA image')
        normalized = transforms.normalize(axial, lower_bound=-400)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(axial, interpolation='none')
        # plt.show()
        file_id = in_blob.name.split('/')[1]
        file_id = file_id.split('.')[0]

        cloud.save_npy_to_cloud(axial, in_blob.name[25:41], 'luke_training')
        logging.info(f'saved .npy file to cloud')

    # from preprocess_luke/validation directory
    for in_blob in bucket.list_blobs(prefix='preprocess_luke/validation/'):
        # blacklist
        if in_blob.name == 'preprocess_luke/validation/LAUIHISOEZIM5ILF.npy':
            continue
        logging.info(f'downloading {in_blob.name}')
        input_arr = cloud.download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")
        transposed_arr = np.transpose(input_arr, (2, 0, 1))
        logging.info(f'transposed to: {transposed_arr.shape}')

        cropped_arr = transforms.crop(transposed_arr, 'luke')
        not_extreme_arr = transforms.remove_extremes(cropped_arr)

        logging.info(f'removed array extremes')
        # create folder w patient ID
        axial = transforms.mip_array(not_extreme_arr, 'axial')
        logging.info(f'mip-ed CTA image')
        normalized = transforms.normalize(axial, lower_bound=-400)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(axial, interpolation='none')
        # plt.show()
        file_id = in_blob.name.split('/')[1]
        file_id = file_id.split('.')[0]

        cloud.save_npy_to_cloud(axial, in_blob.name[27:43], 'luke_validation')
        logging.info(f'saved .npy file to cloud')
