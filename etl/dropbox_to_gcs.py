"""Moves new data in https://www.dropbox.com/home/ELVOs_anon
to gs://elvos/elvos_anon.

To run this code type the following commands in your terminal:
    conda install -c conda-forge dropbox

Then set your google application credentials as explained here:
https://cloud.google.com/docs/authentication/getting-started

Similarly set the environment variable DROPBOX_TOKEN to an
access token.
"""
import os

import dropbox
from dropbox.files import FolderMetadata
from google.cloud import storage

# TODO: Use a config file to store env variables
# to store secret info
dbx = dropbox.Dropbox(os.environ['DROPBOX_TOKEN'])

gcs_client = storage.Client(project='elvo-198322')
bucket = gcs_client.get_bucket('elvos')

results = dbx.files_list_folder('id:ROCtfi_cdqAAAAAAAAB7Uw')

if results.has_more:
    raise RuntimeError('has_more=True is currently not supported')

for entry in results.entries:
    if isinstance(entry, FolderMetadata):
        print('ignoring folder:', entry)
        continue
    blob = storage.Blob('ELVOs_anon/' + entry.name, bucket)
    if blob.exists():
        print('ignoring file already on GCS:', entry.name)
        continue
    print('downloading from dropbox:', entry.name)
    response = dbx.files_download_to_file(entry.name, entry.id)
    print('uploading to GCS:', 'ELVOs_anon/' + entry.name)
    blob.upload_from_filename(entry.name)
    os.remove(entry.name)
