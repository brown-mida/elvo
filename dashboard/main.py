import datetime
import io
import logging
import os

import flask
import gspread
import matplotlib as mpl
import numpy as np
import requests
from google.cloud import storage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from oauth2client.service_account import ServiceAccountCredentials
from skimage import measure

mpl.use('Agg')
from matplotlib import image  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

app = flask.Flask(__name__)

client = storage.Client(project='elvo-198322')
bucket = client.bucket('elvos')

cache = {}

SPREADSHEET_CREDENTIALS = os.environ['SPREADSHEET_CREDENTIALS']

models = [{}]


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/annotator')
def annotator():
    return flask.render_template('annotator.html')


@app.route('/trainer')
def trainer():
    return flask.render_template('trainer.html')


@app.route('/model', methods=['POST'])
def create_model():
    response = requests.post('http://104.196.51.205:8080/api/experimental/dags/'
                             'train_model/dag_runs',
                             json={})

    return response.content, response.status_code, response.headers.items()


@app.route('/roi', methods=['POST'])
def roi():
    data = flask.request.json
    logging.info(f'creating annotation: {data}')
    created_by = data['created_by']
    patient_id = data['patient_id']

    x1 = data['x1']
    x2 = data['x2']
    y1 = data['y1']
    y2 = data['y2']
    z1 = data['z1']
    z2 = data['z2']

    created_at = datetime.datetime.utcnow()

    # TODO: Incorporate db functionality again at
    #  some other point
    # db.session.add(ann)
    # db.session.commit()

    query_string = f'{flask.request.url_root}annotator?' \
                   f'x1={x1}&x2={x2}&y1={y1}&y2={y2}' \
                   f'&z1={z1}&z2={z2}&patientId={patient_id}'
    print(query_string)
    values = [
        patient_id,
        created_by,
        created_at.isoformat(),
        x1,
        x2,
        y1,
        y2,
        z1,
        z2,
        query_string,
    ]

    values.extend([''] * 10)  # Pad so it doesn't shift left.

    # Put here for now since it's not working for everybody
    scope = ['https://spreadsheets.google.com/feeds']
    logging.debug('getting service account credentials')
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        SPREADSHEET_CREDENTIALS,
        scope
    )
    spread_client = gspread.authorize(credentials)
    key = '1_j7mq_VypBxYRWA5Y7ef4mxXqU0EmBKDl0lkp62SsXA'
    worksheet = spread_client.open_by_key(key).worksheet('annotations')
    worksheet.append_row(values)
    logging.info(f'added to spreadsheet: {values}')
    return str('<DATABASE_ID>')


@app.route('/image/dimensions/<patient_id>')
def dimensions(patient_id):
    logging.debug(f'getting dimensions for patient {patient_id}')
    arr = _retrieve_arr(patient_id)
    shape = arr.shape
    logging.debug(f'patient has shape {shape}')

    return flask.json.dumps({
        'z': shape[0],
        'x': shape[1],
        'y': shape[2],
    })


@app.route('/image/axial/<patient_id>/<int:slice_i>')
def axial(patient_id, slice_i):
    arr = _retrieve_arr(patient_id)
    reverse_i = len(arr) - slice_i
    reoriented = np.flip(arr[reverse_i], axis=0)
    return _send_slice(reoriented)


@app.route('/image/axial_mip/<patient_id>/<int:slice_i>')
def axial_mip(patient_id, slice_i):
    arr = _retrieve_arr(patient_id)
    reverse_i = len(arr) - slice_i
    mipped = arr[reverse_i:reverse_i + 24].max(axis=0)
    reoriented = np.flip(mipped, axis=0)
    return _send_slice(reoriented)


@app.route('/image/sagittal/<patient_id>/<int:slice_k>')
def sagittal(patient_id, slice_k):
    arr = _retrieve_arr(patient_id)

    return _send_slice(arr[:, :, slice_k])


@app.route('/image/coronal/<patient_id>/<int:slice_j>')
def coronal(patient_id, slice_j):
    arr = _retrieve_arr(patient_id)
    return _send_slice(arr[:, slice_j, :])


@app.route('/image/rendering/<patient_id>')
def rendering(patient_id):
    x1 = int(flask.request.args.get('x1'))
    x2 = int(flask.request.args.get('x2'))
    y1 = int(flask.request.args.get('y1'))
    y2 = int(flask.request.args.get('y2'))
    z1 = int(flask.request.args.get('z1'))
    z2 = int(flask.request.args.get('z2'))

    arr = _retrieve_arr(patient_id)

    roi = arr[
          min(x1, x2):max(x1, x2),
          min(y1, y2):max(y1, y2),
          min(z1, z2):max(z1, z2)]
    return _send_3d(roi)


def _send_3d(roi: np.ndarray):
    out_stream = io.BytesIO()
    logging.debug('creating 3d rendering')
    _save_3d(out_stream, roi)
    out_stream.seek(0)
    logging.debug('sending 3d rendering')
    return flask.send_file(out_stream,
                           mimetype='image/png')


def _save_3d(file, arr: np.ndarray, threshold=150):
    """
    Saves the array to the file-like object

    :param file: The file-like object to save the array to
    :param arr: The array to save
    :param threshold: Contour value to search for isosurfaces
    :return:
    """

    # Position the scan upright, so the head of the patient would
    # be at the top facing the camera
    p = arr.transpose(2, 1, 0)
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)
    # Fancy indexing: `verts[faces]` to generate a collection
    # of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.savefig(file, format='png')


def _retrieve_arr(patient_id: str) -> np.ndarray:
    cached_arr = cache.get(patient_id)
    if cached_arr is not None:
        logging.debug(f'loading {patient_id} from cache')
        return cached_arr
    logging.debug(f'downloading {patient_id} from GCS')
    blob = bucket.get_blob(f'airflow/npy/{patient_id}.npy')
    if blob is None:
        raise ValueError(
            'Blob with patient id {} does not exist'.format(patient_id))
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)
    arr = np.load(in_stream)
    if len(cache) >= 5:
        cache.popitem()
    cache[patient_id] = arr
    return arr


def _send_slice(arr: np.ndarray):
    out_stream = io.BytesIO()
    image.imsave(out_stream,
                 np.flip(arr, 0),
                 vmin=-200,
                 vmax=400,
                 cmap='gray',
                 format='png')
    out_stream.seek(0)
    return flask.send_file(out_stream,
                           mimetype='image/png')


def validator():
    # Validating data (visually)
    raise NotImplementedError()


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    configure_logger()
    app.run(host='127.0.0.1', port=8080, debug=True)
