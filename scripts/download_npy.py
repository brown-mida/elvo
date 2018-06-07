"""Fetches ELVO key spreadsheet from Google Drive, and saves it in
Google Cloud Storage.

Requires gspread, oauth2client, and google-cloud-storage. To install, run:

pip install gspread oauth2client google-cloud-storage

This requires a `client_secret.json` file in the credentials folder.
Download it (the gcloud service account) from the Google Cloud Console.
"""

import pickle
from google.cloud import storage

gcs_client = storage.Client.from_service_account_json(
    '../credentials/client_secret.json'
)
bucket = gcs_client.get_bucket('elvos')
blob = bucket.get_blob('numpy/ZXXMMCGK6ANRKLFD.npy')
blob.download_to_filename('../tmp/npy/1.npy')
