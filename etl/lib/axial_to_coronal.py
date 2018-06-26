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
        if in_blob.name <= 'numpy/ABPO2BORDNF3OVL3.npy':
            continue

        # blacklist
        if in_blob.name == 'numpy/LAUIHISOEZIM5ILF.npy':
            continue

        # perform the normal MIPing procedure
        logging.info(f'downloading {in_blob.name}')
        axial = cloud.download_array(in_blob)
        coronal = np.transpose(axial, (1, 0, 2))
        coronal = np.fliplr(coronal)

        file_id = in_blob.name.split('/')[1]
        file_id = file_id.split('.')[0]

        try:
            print(f'gs://elvos/mip_data/from_numpy/coronal/{file_id}.npy')
            np.save(file_io.FileIO(f'gs://elvos/mip_data/from_numpy/'
                                   f'coronal/{file_id}.npy', 'w'), coronal)
            np.save(file_io.FileIO(f'gs://elvos/mip_data/from_numpy/'
                                   f'axial/{file_id}.npy', 'w'), axial)
        except Exception as e:
            logging.error(f'for patient ID: {file_id} {e}')
        logging.info(f'saved .npy file to cloud')
