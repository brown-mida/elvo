from main import app


def test_index():
    app.testing = True
    client = app.test_client()

    r = client.get('/')
    assert r.status_code == 200


def test_roi():
    # TODO: Implement this test
    app.testing = True
    client = app.test_client()
    r = client.post('/roi')
