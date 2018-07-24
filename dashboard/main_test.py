import json
import os

import flask
import pytest


@pytest.mark.skipif('SPREADSHEET_CREDENTIALS' not in os.environ,
                    reason='Spreasheet credentials required')
def test_index():
    from main import app  # TODO: Deal with TravisCI failure
    app.testing = True
    client = app.test_client()

    r = client.get('/')
    assert r.status_code == 200


@pytest.mark.skipif('SPREADSHEET_CREDENTIALS' not in os.environ,
                    reason='Spreasheet credentials required')
def test_dimensions():
    from main import app
    app.testing = True
    client = app.test_client()
    r = client.get('/image/dimensions/L66E2921S3O1MURX')
    assert r.status_code == 200


@pytest.mark.skipif('SPREADSHEET_CREDENTIALS' not in os.environ,
                    reason='Spreasheet credentials required')
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


@pytest.mark.skipif('SPREADSHEET_CREDENTIALS' not in os.environ,
                    reason='Spreadsheet credentials required')
def test_create_model():
    from main import app
    app.testing = True
    client = app.test_client()
    data = {}
    r = client.post('/model',
                    data=json.dumps(data),
                    content_type='application/json')
    assert r.status_code == 200


@pytest.mark.skipif('SPREADSHEET_CREDENTIALS' not in os.environ,
                    reason='Spreadsheet credentials required')
def test_list_plots():
    from main import app
    app.testing = True
    client = app.test_client()
    r: flask.Response
    r = client.get('/plots')
    assert r.status_code == 200
    assert 'test_gcs-2018-07-24T17:30:15.191204' in json.loads(r.data)
