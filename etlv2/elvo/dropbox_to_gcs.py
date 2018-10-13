"""Moves new data in https://www.dropbox.com/home/ELVOs_anon
to gs://elvos/ELVOS_anon.

This script will only load top level files and files in
specified subdirectories. The subdirectory files will
be saved in the same GCS folder, meaning this script effectively
flattens those files.

Then set your google application credentials as explained here:
https://cloud.google.com/docs/authentication/getting-started

Similarly set the environment variable DROPBOX_TOKEN to an
access token.
"""
import datetime
import logging

import os

import dropbox
from dropbox.files import FileMetadata, FolderMetadata
from google.cloud import storage


def upload_entry(blob, dbx, entry):
    logging.info(f'downloading from dropbox: {entry.name}')
    dbx.files_download_to_file(entry.name, entry.id)
    logging.info(f'uploading to GCS: gs://ELVOs_anon/{entry.name}')
    blob.upload_from_filename(entry.name)
    os.remove(entry.name)


def upload_entry_if_outdated(entry: FileMetadata,
                             dbx: dropbox.Dropbox,
                             bucket: storage.Bucket) -> None:
    """Saves the dropbox entry in GCS as
    gs://elvos/ELVOs_anon/{entry.name} if the corresponding
    GCS file is outdated.

    We compare using last updated dates.
    """
    blob = storage.Blob('ELVOs_anon/' + entry.name, bucket)

    if blob.exists():
        blob.reload()  # Needed otherwise blob.updated is None
        dropbox_time = entry.server_modified.replace(
            tzinfo=datetime.timezone.utc)
    else:
        dropbox_time = None

    if not dropbox_time:
        logging.info('blob {} does not exist, inserting'.format(blob.name))
        upload_entry(blob, dbx, entry)
    elif dropbox_time >= blob.updated:
        logging.info('blob {} is outdated, updating'.format(blob.name))
        upload_entry(blob, dbx, entry)
    else:
        logging.info('blob {} is not outdated, not uploading'.format(blob.name))


if __name__ == '__main__':
    dbx = dropbox.Dropbox(os.environ['DROPBOX_TOKEN'])

    gcs_client = storage.Client(project='elvo-198322')
    bucket = gcs_client.get_bucket('elvos')

    # This is the ELVOs_anon folder on Dropbox (10/13/2018).
    # You will need permission from Matt Stib to read from this folder.
    results = dbx.files_list_folder('id:ROCtfi_cdqAAAAAAAAB7Uw')

    if results.has_more:
        raise RuntimeError('has_more=True is currently not supported')

    for entry in results.entries:
        # TODO(#102): Explain '7_11 Redownloaded Studies'
        if entry.name in ('7_17 New ELVOs'):
            logging.info('uploading files in folder: {}'.format(entry.name))
            subdir_results = dbx.files_list_folder(entry.id)
            for e in subdir_results.entries:
                upload_entry_if_outdated(e, dbx, bucket)
        elif isinstance(entry, FolderMetadata):
            logging.info('ignoring folder: {}'.format(entry.name))
        else:
            upload_entry_if_outdated(entry, dbx, bucket)
