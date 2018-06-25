import io
import logging
import os

import flask
import matplotlib as mpl
import numpy as np

mpl.use('Agg')
from google.cloud import storage
from matplotlib import image
from matplotlib import pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

app = flask.Flask(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'elvo-7136c1299dea.json'

client = storage.Client(project='elvo-198322')
bucket = client.bucket('elvos')


# blob = bucket.get_blob(f'numpy/0DQO9A6UXUQHR8RA.npy')
# blob.download_to_filename('tmp.npy')


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # TODO: Change level
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


configure_logger()


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/annotator')
def annotator():
    return flask.render_template('annotator.html')


@app.route('/image/dimensions/<patient_id>/')
def dimensions(patient_id):
    arr = _download_arr(patient_id)
    shape = arr.shape
    return flask.json.dumps({
        'z': shape[0],
        'x': shape[1],
        'y': shape[2],
    })


@app.route('/image/axial/<patient_id>/<int:slice_i>')
def axial(patient_id, slice_i):
    arr = _download_arr(patient_id)
    out_stream = io.BytesIO()
    image.imsave(out_stream,
                 arr[slice_i],
                 vmin=-200,
                 vmax=400,
                 cmap='gray',
                 format='png')
    out_stream.seek(0)
    return flask.send_file(out_stream,
                           mimetype='image/png')


@app.route('/image/axial_mip/<patient_id>/<int:slice_i>')
def axial_mip(patient_id, slice_i):
    arr = _download_arr(patient_id)
    out_stream = io.BytesIO()
    image.imsave(out_stream,
                 arr[slice_i:slice_i + 24].max(axis=0),
                 vmin=-200,
                 vmax=400,
                 cmap='gray',
                 format='png')
    out_stream.seek(0)
    return flask.send_file(out_stream,
                           mimetype='image/png')


@app.route('/image/sagittal/<patient_id>/<int:slice_k>')
def sagittal(patient_id, slice_k):
    arr = _download_arr(patient_id)
    out_stream = io.BytesIO()
    image.imsave(out_stream,
                 np.flip(arr[:, :, slice_k], 0),
                 vmin=-200,
                 vmax=400,
                 cmap='gray',
                 format='png')
    out_stream.seek(0)
    return flask.send_file(out_stream,
                           mimetype='image/png')


@app.route('/image/coronal/<patient_id>/<slice_j>')
def coronal(patient_id, slice_j):
    arr = _download_arr(patient_id)
    out_stream = io.BytesIO()
    image.imsave(out_stream,
                 np.flip(arr[:, :, slice_j], 0),
                 vmin=-200,
                 vmax=400,
                 cmap='gray',
                 format='png')
    out_stream.seek(0)
    return flask.send_file(out_stream,
                           mimetype='image/png')


@app.route('/image/rendering/<patient_id>/<threshold>')
def rendering(patient_id, threshold):
    x1 = flask.request.args.get('x1')
    x2 = flask.request.args.get('x2')
    y1 = flask.request.args.get('y1')
    y2 = flask.request.args.get('y2')
    z1 = flask.request.args.get('z1')
    z2 = flask.request.args.get('z2')

    arr = _download_arr(patient_id)
    out_stream = io.BytesIO()
    logging.debug('creating 3d rendering')
    roi = arr[min(x1, x2):max(x1, x2),
          min(y1, y2):max(y1, y2),
          min(z1, z2):max(z1, z2)]
    save_3d(out_stream, roi, threshold=threshold)
    out_stream.seek(0)
    logging.debug('sending 3d rendering')
    return flask.send_file(out_stream,
                           mimetype='image/png')


def save_3d(file, arr, threshold=150):
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


def _download_arr(patient_id: str) -> np.ndarray:
    blob = bucket.get_blob(f'numpy/{patient_id}.npy')
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)
    return np.load(in_stream)


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
