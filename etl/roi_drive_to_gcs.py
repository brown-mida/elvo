import csv
import logging
import os

import gspread
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials


def roi_to_gcs():
    """Loads the google cloud spreadsheet to GCS.
    """
    logging.info('authenticating')
    scope = ['https://spreadsheets.google.com/feeds']

    creds = ServiceAccountCredentials.from_json_keyfile_name(
        os.environ['DRIVE_KEYFILE'],
        scope
    )
    gspread_client = gspread.authorize(creds)

    spreadsheet = gspread_client.open_by_key(
        '1_j7mq_VypBxYRWA5Y7ef4mxXqU0EmBKDl0lkp62SsXA')
    worksheet = spreadsheet.get_worksheet(0)
    records = worksheet.get_all_records()

    with open('/tmp/annotations.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    gcs_client = storage.Client.from_service_account_json(
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    )
    bucket = gcs_client.get_bucket('elvos')
    blob = storage.Blob('airflow/annotations.csv', bucket)
    blob.upload_from_filename('/tmp/annotations.csv')


if __name__ == '__main__':
    roi_to_gcs()
