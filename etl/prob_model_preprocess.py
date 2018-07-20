import logging
# from ml.models.three_d import c3d
from keras.models import load_model
import numpy as np
import pandas as pd
from lib import cloud_management as cloud


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_prob_scores():
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')
    # model = c3d.C3DBuilder.build()
    model = load_model('tmp/c3d_100.hdf5')
    labels = pd.read_csv('/Users/haltriedman/Desktop/annotated_labels.csv')
    print(labels)

    # loop through every array on GCS
    for in_blob in bucket.list_blobs(prefix='airflow/npy'):
        # blacklist
        if in_blob.name == 'airflow/npy/LAUIHISOEZIM5ILF.npy':
            continue

        # get the file id
        file_id = in_blob.name.split('/')[2]
        file_id = file_id.split('.')[0]
        arr = cloud.download_array(in_blob)
        chunks = []

        for i in range(0, len(arr), 32):
            for j in range(0, len(arr[0]), 32):
                for k in range(0, len(arr[0][0]), 32):
                    # copy the chunk
                    chunk = arr[i:(i + 32), j:(j + 32), k:(k + 32)]
                    # calculate the airspace
                    airspace = np.where(chunk < -300)
                    # if it's less than 90% airspace
                    if (airspace[0].size / chunk.size) < 0.9:
                        chunks.append(chunk)

        preds = model.predict(chunks, batch_size=16)
        cloud.save_preds_to_cloud(preds, file_id)


def main():
    configure_logger()
    get_prob_scores()


if __name__ == '__main__':
    main()
