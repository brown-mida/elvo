"""Fetches ELVO key spreadsheet from Google Drive, and saves it in
Google Cloud Storage.

This requires a `client_secret.json` file in the credentials folder.
Download it (the gcloud service account) from the Google Cloud Console.
"""

import csv
import os

import gspread
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials

# Authenticate to Google Drive
print("Authenticating to Google Drive...")
scope = ['https://spreadsheets.google.com/feeds']
# TODO: Credentials should be kept in secrets/
creds = ServiceAccountCredentials.from_json_keyfile_name(
    '../credentials/client_secret.json', scope
)
client = gspread.authorize(creds)

# Get the Google Spreadsheet data
print("Opening spreadsheet...")
# sheet = client.open_by_key("1hndSmw8dxQVp1d8Fohb8yLOq7_PuCMKlyVS1-6PzLl4")
sheet = client.open_by_key("1ho920hPecJY3pL65_IMXlU9ggqFntFMnZihnMT3h8gs")

print("Get positive data...")
sheet_pos = sheet.get_worksheet(0)
data = sheet_pos.get_all_records()
keys = data[0].keys()
with open('elvo_keys_positive.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)

print("Get negative data...")
sheet_neg = sheet.get_worksheet(1)
data = sheet_neg.get_all_records()
keys = data[0].keys()
with open('elvo_keys_negative.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)

# Upload to Google Cloud
print("Uploading to Google Cloud...")
gcs_client = storage.Client.from_service_account_json(
    '../credentials/client_secret.json'
)
bucket = gcs_client.get_bucket('elvos')

blob = storage.Blob('metadata/positives.csv', bucket)
blob.upload_from_filename('elvo_keys_positive.csv')
blob = storage.Blob('metadata/negatives.csv', bucket)
blob.upload_from_filename('elvo_keys_negative.csv')

# Cleanup
os.remove('elvo_keys_positive.csv')
os.remove('elvo_keys_negative.csv')
print("Done.")
