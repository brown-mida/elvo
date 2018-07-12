import json
import os

import pytest


@pytest.mark.skipif('TRAVIS' in os.environ, reason='Database not on travis')
def test_index():
    from main import app  # TODO: Deal with TravisCI failure
    app.testing = True
    client = app.test_client()

    r = client.get('/')
    assert r.status_code == 200


@pytest.mark.skipif('TRAVIS' in os.environ, reason='Database not on travis')
def test_roi():
    from main import app
    app.testing = True
    client = app.test_client()
    data = {
        'patient_id': 'abc',
        'created_by': 'pytest',
        'x1': -1,
        'x2': -1,
        'y1': -1,
        'y2': -1,
        'z1': -1,
        'z2': -1,
    }
    r = client.post('/roi',
                    data=json.dumps(data),
                    content_type='application/json')
    assert r.status_code == 200
