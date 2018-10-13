import dropbox
import os
import pytest
from google.cloud import storage

from elvo import dropbox_to_gcs


@pytest.mark.skipif('DROPBOX_TOKEN' not in os.environ,
                    reason='Token needed for the test')
def test_upload_entry_if_outdated():
    dbx = dropbox.Dropbox(os.environ['DROPBOX_TOKEN'])

    gcs_client = storage.Client(project='elvo-198322')
    bucket = gcs_client.get_bucket('elvos')

    entry = dbx.files_get_metadata('/ELVOs_anon/068WBWCQGW5JHBYV.cab')
    dropbox_to_gcs.upload_entry_if_outdated(entry, dbx, bucket)
