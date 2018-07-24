import os

import pytest
from google.cloud import storage

from processed_dag import to_public_png


@pytest.mark.skipif('TRAVIS' in os.environ,
                    reason='Requires google credentials')
def test_to_png():
    gcs_client = storage.Client(project='elvo-198322')
    in_bucket = gcs_client.get_bucket('elvos')
    blob = in_bucket.get_blob('processed/processed-lower/'
                              'arrays/0KSBX96F8BU1FCFQ.npy')
    out_bucket = gcs_client.get_bucket('elvos-public')
    to_public_png(blob, out_bucket)
