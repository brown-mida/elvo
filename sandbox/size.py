import os
import csv
import random
import numpy as np
from scipy.ndimage.interpolation import zoom

from google.cloud import storage

# Get npy files from Google Cloud Storage
gcs_client = storage.Client.from_service_account_json(
    'credentials/client_secret.json'
)
bucket = gcs_client.get_bucket('elvos')
blob = bucket.get_blob('preprocess_luke/training/04IOS24JP70LHBGB.npy')
print(blob)
blob.download_to_filename('tmp/npy/aaa.npy')
arr = np.load('tmp/npy/aaa.npy')
print(np.shape(arr))
