from lib import roi_transforms, transforms, cloud_management as cloud
import logging
import numpy as np


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def create_chunks():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    for in_blob in bucket.list_blobs(prefix='airflow/npy'):
        # blacklist
        if in_blob.name == 'airflow/npy/LAUIHISOEZIM5ILF.npy':
            continue

        file_id = in_blob.name.split('/')[2]
        file_id = file_id.split('.')[0]

        arr = cloud.download_array(in_blob)
        arr = np.transpose(arr, (1, 2, 0))
        arr = transforms.segment_vessels(arr)
        # print(arr.shape)
        arr = roi_transforms.convert_multiple_32(arr)
        # print(arr.shape)

        chunks = []
        for i in range(0, len(arr), 32):
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    print(i, j, k)
                    chunk = arr[i:i+32][j:j+32][k:k+32]
                    # print(file_id)
                    print(chunk.shape)

                    airspace = np.where(chunk == -50)
                    if (len(airspace) / chunk.size) < 0.9:
                        chunks.append(chunk.tolist())

        chunks = np.ndarray(chunks)
        print(chunks.shape)


if __name__ == '__main__':
    create_chunks()
