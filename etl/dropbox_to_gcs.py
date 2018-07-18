"""Moves new data in https://www.dropbox.com/home/ELVOs_anon
to gs://elvos/elvos_anon.

To run this code type the following commands in your terminal:
    conda install -c conda-forge dropbox

Then set your google application credentials as explained here:
https://cloud.google.com/docs/authentication/getting-started

Similarly set the environment variable DROPBOX_TOKEN to an
access token.
"""
import datetime

import dropbox
import os
from dropbox.files import FileMetadata, FolderMetadata
from google.cloud import storage


def upload_entry_if_outdated(entry: FileMetadata,
                             dbx: dropbox.Dropbox,
                             bucket: storage.Bucket):
    """Uploads the entry to Google Cloud"""
    blob = storage.Blob('ELVOs_anon/' + entry.name, bucket)
    blob.reload()
    dropbox_time = entry.server_modified.replace(tzinfo=datetime.timezone.utc)
    if dropbox_time < blob.updated:
        print('blob {} is not outdated, not uploading'.format(blob.name))
    else:
        print('downloading from dropbox:', entry.name)
        dbx.files_download_to_file(entry.name, entry.id)
        print('uploading to GCS:', 'ELVOs_anon/' + entry.name)
        blob.upload_from_filename(entry.name)
        os.remove(entry.name)


if __name__ == '__main__':
    dbx = dropbox.Dropbox(os.environ['DROPBOX_TOKEN'])

    gcs_client = storage.Client(project='elvo-198322')
    bucket = gcs_client.get_bucket('elvos')

    results = dbx.files_list_folder('id:ROCtfi_cdqAAAAAAAAB7Uw')

    if results.has_more:
        raise RuntimeError('has_more=True is currently not supported')

    for entry in results.entries:
        if isinstance(entry, FolderMetadata):
            print('ignoring folder:', entry)
        else:
            upload_entry_if_outdated(entry, dbx, bucket)
