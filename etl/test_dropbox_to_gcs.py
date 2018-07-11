import os

import dropbox
import pytest
from google.cloud import storage

import dropbox_to_gcs


@pytest.mark.skipif('DROPBOX_TOKEN' not in os.environ,
                    reason='Token needed for the test')
def test_upload_entry():
    dbx = dropbox.Dropbox(os.environ['DROPBOX_TOKEN'])

    gcs_client = storage.Client(project='elvo-198322')
    bucket = gcs_client.get_bucket('elvos')

    entry = dbx.files_get_metadata('/ELVOs_anon/068WBWCQGW5JHBYV.cab')
    dropbox_to_gcs.upload_entry(entry, dbx, bucket)
