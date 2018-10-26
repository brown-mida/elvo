import io
import logging

import pandas as pd
from google.cloud import storage


def load_metadata(bucket: storage.Bucket) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads metadata/positives.csv and metadata/negatives.csv from GCS.

    :param bucket: the GCS bucket, typically 'elvos'
    :return: positives_df, negatives_df
    """
    positives_blob = bucket.get_blob('metadata/positives.csv')
    positives_bytes = positives_blob.download_as_string()
    positives_str = positives_bytes.decode('utf-8')
    positives_df = pd.read_csv(io.StringIO(positives_str))

    negatives_blob = bucket.get_blob('metadata/negatives.csv')
    negatives_bytes = negatives_blob.download_as_string()
    negatives_str = negatives_bytes.decode('utf-8')
    negatives_df = pd.read_csv(io.StringIO(negatives_str))
    return positives_df, negatives_df


def create_labels_csv(positives_df, negatives_df, bucket, in_dir):
    """
    Constructs the gs://elvos/processed/labels.csv file from the
    DataFrame and the .cab/.zip data on GCS.

    labels.csv has 2 columns:
     - patient_id: the patient Anon ID
     - label: 1 or 0

    :param positives_df:
    :param negatives_df:
    :param bucket:
    :param in_dir: directory containing .npy data
    :return:
    """
    labels = []
    blob: storage.Blob
    for blob in bucket.list_blobs(prefix=in_dir):
        if len(blob.name) < 4 or blob.name[-4:] not in ('.zip', '.cab'):
            logging.info(f'ignoring non-data file {blob.name}')
            continue

        patient_id = blob.name[len(in_dir): -len('.npy')]

        if patient_id in positives_df['Anon ID'].values:
            labels.append((patient_id, 1))
        elif patient_id in negatives_df['Anon ID'].values:
            labels.append((patient_id, 0))
        else:
            msg = f'blob with patient id {patient_id} not found in' \
                  f' the metadata CSVs'
            logging.error(msg)
            raise ValueError(msg)

    labels_df = pd.DataFrame(labels, columns=['patient_id', 'label'])
    labels_df.to_csv('/tmp/labels.csv', index=False)
    logging.info('uploading labels to the gs://elvos/processed/labels.csv')
    labels_blob = storage.Blob('processed/labels.csv', bucket=bucket)
    labels_blob.upload_from_filename('/tmp/labels.csv')
    logging.info(f'label value counts {labels_df["label"].value_counts()}')


def prepare_labels(in_dir: str):
    gcs_client = storage.Client(project='elvo-198322')
    input_bucket = gcs_client.get_bucket('elvos')

    positives_df, negatives_df = load_metadata(input_bucket)
    create_labels_csv(positives_df, negatives_df, input_bucket, in_dir)
