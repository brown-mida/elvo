import logging
import numpy as np
# from matplotlib import pyplot as plt
from tensorflow.python.lib.io import file_io
import cloud_management as cloud


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

    for in_blob in bucket.list_blobs(prefix='numpy'):

        # blacklist
        if in_blob.name == 'numpy/LAUIHISOEZIM5ILF.npy':
            continue
        elif in_blob.name == 'numpy/ALOUY4SF3BQKXQCZ.npy':
            continue
        elif in_blob.name == 'numpy/ABPO2BORDNF3OVL3.npy':
            continue

        # perform the normal MIPing procedure
        logging.info(f'downloading {in_blob.name}')
        axial = cloud.download_array(in_blob)
        coronal = np.transpose(axial, (1, 0, 2))
        coronal = np.fliplr(coronal)
        sagittal = np.transpose(axial, (2, 0, 1))
        sagittal = np.fliplr(sagittal)

        file_id = in_blob.name.split('/')[1]
        file_id = file_id.split('.')[0]

        try:
            coronal_io = file_io.FileIO(f'gs://elvos/numpy/coronal/'
                                        f'{file_id}.npy', 'w')
            np.save(coronal_io, coronal)
            axial_io = file_io.FileIO(f'gs://elvos/numpy/axial/'
                                      f'{file_id}.npy', 'w')
            np.save(axial_io, axial)
            sagittal_io = file_io.FileIO(f'gs://elvos/numpy/sagittal/'
                                         f'{file_id}.npy', 'w')
            np.save(sagittal_io, sagittal)
            coronal_io.close()
            axial_io.close()
            sagittal_io.close()

        except Exception as e:
            logging.error(f'for patient ID: {file_id} {e}')
            break
        logging.info(f'saved .npy file to cloud')
