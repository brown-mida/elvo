import flask
import requests
from google.cloud import storage

app_train = flask.Blueprint('app_train', __name__)

client = storage.Client(project='elvo-198322')
# TODO(luke): Start splitting the development/private bucket from the public
# TODO(luke): Also start considering user-specific buckets.
pub_bucket = client.bucket('elvos-public')


@app_train.route('/trainer')
def trainer():
    return flask.render_template('trainer.html')


@app_train.route('/model', methods=['POST'])
def create_model():
    data = flask.json.dumps(flask.request.get_json())
    response = requests.post('http://104.196.51.205:8080/api/experimental/'
                             'dags/train_model/dag_runs',
                             json={'conf': data})
    return response.content, response.status_code, response.headers.items()


@app_train.route('/plots')
def list_plots():
    """
    Gets all available plots.

    :return: a JSON object containing the gs urls of the available plots.
    """
    plots = set()

    blob: storage.Blob
    for blob in pub_bucket.list_blobs(prefix='plots/'):
        plot_dir = blob.name.split('/')[1]
        if not plot_dir.startswith('test_'):
            plots.add(plot_dir)
    return flask.json.jsonify(list(plots))


@app_train.route('/data')
def list_datasets():
    """
    Gets all available datasets.

    :return: a JSON object containing the gs urls of the available plots.
    """
    datasets = set()

    blob: storage.Blob
    for blob in pub_bucket.list_blobs(prefix='processed/'):
        data_name = blob.name.split('/')[1]
        if not data_name.startswith('test'):
            datasets.add(data_name)
    return flask.json.jsonify(list(datasets))


@app_train.route('/data/<image_name>')
def get_images(image_name: str):
    """
    Returns the image names (ex. 0ABCSDFS.png) which have the given data name

    :return:
    """
    images = []

    blob: storage.Blob
    for blob in pub_bucket.list_blobs(prefix=f'processed/{image_name}/'):
        image_name = blob.name.split('/')[-1]
        images.append(image_name)
    return flask.json.jsonify(list(images))
