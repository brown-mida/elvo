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
        arr = roi_transforms.convert_multiple_32(arr)
        stripped = transforms.segment_vessels(arr)
        point_cloud = transforms.point_cloud(arr)

        chunks = []
        stripped_chunks = []
        pc_chunks = []
        for i in range(0, len(arr), 32):
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):

                    chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                    airspace = np.where(chunk < -300)
                    if (airspace[0].size / chunk.size) < 0.9:
                        chunks.append(chunk.tolist())

                    stripped_chunk = stripped[i:(i + 32), j:(j + 32), k:(k + 32)]
                    stripped_airspace = np.where(stripped_chunk <= -50)
                    if (stripped_airspace[0].size / stripped_chunk.size) < 0.9:
                        stripped_chunks.append(stripped_chunk.tolist())

                    pc_chunk = point_cloud[i:(i + 32), j:(j + 32), k:(k + 32)]
                    pc_airspace = np.where(pc_chunk == 0)
                    if (pc_airspace[0].size / pc_chunk.size) < 0.9:
                        pc_chunks.append(pc_chunk.tolist())

        np_chunks = np.asarray(chunks)
        np_stripped_chunks = np.asarray(stripped_chunks)
        np_pc_chunks = np.asarray(pc_chunks)
        cloud.save_chunks_to_cloud(np_chunks, 'normal', file_id)
        cloud.save_chunks_to_cloud(np_stripped_chunks, 'stripped', file_id)
        cloud.save_chunks_to_cloud(np_pc_chunks, 'point_cloud', file_id)


if __name__ == '__main__':
    create_chunks()
