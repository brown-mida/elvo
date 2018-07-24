import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib import cloud_management as cloud  # , roi_transforms, transforms


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def process_labels():
    positives_df = pd.read_csv(
        '/Users/amy/Documents/ELVOproject/ELVO_Annotator_Key_Positives.csv')
    # print(positives_df['0', '1', '2', '3', '4'])
    positives_df.drop([0, 1, 2])
    positives_df.drop(positives_df.columns[0], axis=1)
    positives_df = positives_df.filter(items=['Unnamed: 0',
                                                  'Unnamed: 3'])
    for index, row in positives_df.iterrows():
        print(index, row)

    logging.info(positives_df)

    negatives_df = pd.read_csv(
        '/Users/amy/Documents/ELVOproject/ELVO_Annotator_Key_Negatives.csv')
    logging.info(negatives_df)

    to_add = {}
    for index, row in negatives_df.iterrows():
        patient_id = row[0]
        #(m1, m2, m3, ica, basilar, vert)
        to_add[patient_id] = str((0, 0, 0, 0, 0, 0))

    for index, row in positives_df.iterrows():
        patient_id = row[0]
        m1 = 0
        m2 = 0
        m3 = 0
        ica = 0
        basilar = 0
        vert = 0
        print("unnamed 0: " + str(row['Unnamed: 0'])
                + "unnamed 3: " + str(row['Unnamed: 3']))
        if '**' in str(row['Unnamed: 0']) \
                or 'Anon' in str(row['Unnamed: 0']) \
                or row['Unnamed: 0'] == '':
            continue
        if 'M1' in str(row['Unnamed: 3']):
            m1 = 1
        if 'M2' in str(row['Unnamed: 3']):
            m2 = 1
        if 'M3' in str(row['Unnamed: 3']):
            m3 = 1
        if 'ICA' in str(row['Unnamed: 3']):
            ica = 1
        # use 'asilar' to account for capitalization
        if 'asilar' in str(row['Unnamed: 3']):
            basilar = 1
        if 'vert' in str(row['Unnamed: 3']):
            vert = 1
        to_add[patient_id] = str((m1, m2, m3, ica, basilar, vert))

    to_add_df = pd.DataFrame.from_dict(
        to_add, orient='index', columns=['Unnamed: 3'])
    logging.info(to_add_df)

    to_add_df.to_csv(
        '/Users/amy/Documents/ELVOproject/classification_vectors.csv')
    print("Saving file")


def run_preprocess():
    configure_logger()
    process_labels()

if __name__ == '__main__':
    run_preprocess()
