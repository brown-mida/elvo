import io
import logging

import pandas as pd
from google.cloud import storage

IN_DIR = 'ELVOs_anon/'


def load_metadata(bucket):
    """Loads the metadata from GCS.
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


def create_labels_csv(bucket, positives_df, negatives_df) -> None:
    """Creates a file labels.csv mapping patient_id to 1, if positive
    and 0, if negative.
    """
    labels = []
    blob: storage.Blob
    for blob in bucket.list_blobs(prefix=IN_DIR):
        if blob.name.endswith('.csv'):
            continue  # Ignore the metadata CSV

        patient_id = blob.name[len(IN_DIR): -len('.npy')]

        if patient_id in positives_df['Anon ID'].values:
            labels.append((patient_id, 1))
        elif patient_id in negatives_df['Anon ID'].values:
            labels.append((patient_id, 0))
        else:
            logging.warning(f'blob with patient id {patient_id} not found in'
                            f' the metadata CSVs, defaulting to negative')
            labels.append((patient_id, 0))
    labels_df = pd.DataFrame(labels, columns=['patient_id', 'label'])
    labels_df.to_csv('labels.csv', index=False)
    logging.info('uploading labels to the gs://elvos/processed/labels.csv')
    labels_blob = storage.Blob('processed/labels.csv', bucket=bucket)
    labels_blob.upload_from_filename('labels.csv')
    logging.info(f'label value counts {labels_df["label"].value_counts()}')


if __name__ == '__main__':
    gcs_client = storage.Client(project='elvo-198322')
    input_bucket = gcs_client.get_bucket('elvos')

    positives_df, negatives_df = load_metadata(input_bucket)
    create_labels_csv(input_bucket, positives_df, negatives_df)
