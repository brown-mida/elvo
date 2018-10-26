import dropbox
import os
import pytest
from google.cloud import storage

from .dropbox_to_gcs import upload_entry_if_outdated


@pytest.mark.skipif('DROPBOX_TOKEN' not in os.environ,
                    reason='Token needed for the test')
def test_upload_entry_if_outdated():
    dbx = dropbox.Dropbox(os.environ['DROPBOX_TOKEN'])

    gcs_client = storage.Client(project='elvo-198322')
    bucket = gcs_client.get_bucket('elvos')

    entry = dbx.files_get_metadata('/ELVOs_anon/068WBWCQGW5JHBYV.cab')
    upload_entry_if_outdated(entry, dbx, bucket)
