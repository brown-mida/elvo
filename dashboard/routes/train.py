"""
Responsible for the routes related to the /trainer endpoint.

"""
import io
import logging
import types
from concurrent.futures import ThreadPoolExecutor

import flask
import matplotlib
import numpy as np
import requests
from google.cloud import storage

import blueno

matplotlib.use('agg')
from matplotlib import pyplot as plt  # noqa: E402

app_train = flask.Blueprint('app_train', __name__)

client = storage.Client(project='elvo-198322')
# TODO(luke): Start splitting the development/private bucket from the public
# TODO(luke): Also start considering user-specific buckets.
priv_bucket = client.bucket('elvos')
pub_bucket = client.bucket('elvos-public')

executor = ThreadPoolExecutor(2)


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


@app_train.route('/data/<data_name>')
def get_images(data_name: str):
    """
    Returns the image names (ex. 0ABCSDFS.png) which have the given data name

    :return:
    """
    images = []

    blob: storage.Blob
    for blob in pub_bucket.list_blobs(prefix=f'processed/{data_name}/'):
        data_name = blob.name.split('/')[-1]
        images.append(data_name)
    return flask.json.jsonify(list(images))


@app_train.route('/preprocessing/transforms', methods=['GET'])
def list_transforms():
    """
    Returns transforms in blueno.transforms

    :return: all functions in blueno.transforms
    """
    transforms = [attr_name for attr_name in dir(blueno.transforms)
                  if isinstance(getattr(blueno.transforms, attr_name),
                                types.FunctionType)]
    return flask.json.jsonify(transforms)


@app_train.route('/preprocessing/<data_name>', methods=['POST'])
def preprocess_data(data_name: str):
    """
    Processes the images using the preprocessing functions defined in
    the payload, saving the output in gs://elvos-public/processed/data_name.

    :param data_name:
    :return:
    """
    # TODO(luke): Since this requests takes so long it may be better to do this
    # operation on a different server.
    data = flask.request.get_json()
    # TODO(luke): Validate this so it fits in ResNet
    crop_length = int(data['cropLength'])
    mip_thickness = int(data['mipThickness'])
    height_offset = int(data['heightOffset'])
    pixel_value_range = (int(data['minPixelValue']),
                         int(data['maxPixelValue']))

    input_blob: storage.Blob
    executor.submit(_process_arrays,
                    data_name, crop_length, height_offset, mip_thickness,
                    pixel_value_range)

    return '', 204  # No Content code


def _process_arrays(data_name, crop_length, height_offset,
                    mip_thickness, pixel_value_range):
    for input_blob in priv_bucket.list_blobs(prefix=f'airflow/npy'):
        logging.info(f'downloading numpy file: {input_blob.name}')
        input_filename = input_blob.name.split('/')[-1]
        input_stream = io.BytesIO()
        input_blob.download_to_file(input_stream)
        input_stream.seek(0)
        arr = np.load(input_stream)

        logging.debug(f'processing numpy array')
        try:
            arr = _process_arr(arr, crop_length, height_offset,
                               mip_thickness, pixel_value_range)
        except AssertionError:
            logging.debug(f'file {input_filename} could not be processed,'
                          f' has input shape {arr.shape}')

        logging.debug(f'converting processed array into png')
        output_stream = io.BytesIO()
        # Standardize to [0, 1] otherwise it fails
        # TODO(luke): Consider standardizing by feature (RGB) as that is how
        # Keras does it
        standardized = (arr - arr.min()) / arr.max()
        plt.imsave(output_stream, standardized)
        output_stream.seek(0)

        output_filename = input_filename.replace('.npy', '.png')
        output_blob = pub_bucket.blob(
            f'processed/{data_name}/arrays/{output_filename}')

        logging.info(f'uploading png file: {output_blob.name}')
        output_blob.upload_from_file(output_stream)


def _process_arr(arr,
                 crop_length,
                 height_offset,
                 mip_thickness,
                 pixel_value_range):
    arr = blueno.transforms.crop(arr,
                                 (3 * mip_thickness,
                                  crop_length,
                                  crop_length),
                                 height_offset=height_offset)
    arr = np.stack([arr[:mip_thickness],
                    arr[mip_thickness: 2 * mip_thickness],
                    arr[2 * mip_thickness: 3 * mip_thickness]])
    arr = blueno.transforms.bound_pixels(arr,
                                         pixel_value_range[0],
                                         pixel_value_range[1])
    arr = arr.max(axis=1)
    arr = arr.transpose((1, 2, 0))
    return arr
