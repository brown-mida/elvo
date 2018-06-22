import os
from google.cloud import storage

# Delete all content in tmp/npy/
filelist = [f for f in os.listdir('../tmp/npy')]
for f in filelist:
    os.remove(os.path.join('../tmp/npy', f))

# Get npy files from Google Cloud Storage
gcs_client = storage.Client.from_service_account_json(
    '../credentials/client_secret.json'
)
bucket = gcs_client.get_bucket('elvos')
blobs = bucket.list_blobs(prefix='mip_data/from_numpy/')

for blob in blobs:
    file = blob.name
    id = file.split('/')[-1].split('.')[0]
    blob.download_to_filename('../tmp/npy/{}.npy'.format(id))
    print(id)
