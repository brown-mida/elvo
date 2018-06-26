import logging
import numpy as np
from matplotlib import pyplot as plt
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

        # perform the normal MIPing procedure
        logging.info(f'downloading {in_blob.name}')
        axial = cloud.download_array(in_blob)
        sagittal = np.transpose(axial, (2, 0, 1))
        sagittal = np.fliplr(sagittal)

        file_id = in_blob.name.split('/')[1]
        file_id = file_id.split('.')[0]

        for i in range(0, len(sagittal), 10):
            plt.figure(figsize=(6, 6))
            plt.imshow(sagittal[i], interpolation='none')
            plt.show()

        try:
            print(f'gs://elvos/mip_data/from_numpy/sagittal/{file_id}.npy')
            np.save(file_io.FileIO(f'gs://elvos/mip_data/from_numpy/'
                                   f'sagittal/{file_id}.npy', 'w'), sagittal)
        except Exception as e:
            logging.error(f'for patient ID: {file_id} {e}')
        logging.info(f'saved .npy file to cloud')
