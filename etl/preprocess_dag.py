import datetime
import logging

import io
import matplotlib
import numpy as np
from airflow.models import BaseOperator, DAG
from google.cloud import storage

import blueno

matplotlib.use('agg')
from matplotlib import pyplot as plt  # noqa: E402


class PreprocessOperator(BaseOperator):
    """
    Preprocess the image data, using the transforms specified in the
    context
    """

    def execute(self, context):
        data = context['dag_run'].conf
        logging.info('configuration is: {}'.format(data))

        client = storage.Client(project='elvo-198322')
        data_bucket = client.bucket('elvos')
        image_bucket = client.bucket('elvos-public')

        # TODO(luke): Validate this so it fits in ResNet
        data_name = data['data_name']
        crop_length = int(data['cropLength'])
        mip_thickness = int(data['mipThickness'])
        height_offset = int(data['heightOffset'])
        pixel_value_range = (int(data['minPixelValue']),
                             int(data['maxPixelValue']))
        process_arrays(data_name,
                       crop_length,
                       height_offset,
                       mip_thickness,
                       pixel_value_range,
                       data_bucket,
                       image_bucket)


def process_arrays(data_name, crop_length, height_offset,
                   mip_thickness, pixel_value_range, data_bucket,
                   image_bucket):
    """
    Processes the arrays, loading PNGS to
    gs://elvos-public/processed/{data_name}/arrays/ and npy files to
    gs://elvos/processed/{data_name}/arrays

    TODO(luke): Code to train on data only available on GCS (aka download
    in bluenot.py).

    :param data_name: the directory name of the data.
    :param crop_length: the length (and width) of an image
    :param height_offset: the offset from the top of the image
        (above the head) to start mipping from.
    :param mip_thickness: the thickness of one of the 3 mip slices
    :param pixel_value_range: the range of values to bound the CT values at
    :param data_bucket: the bucket to store the data in, currently this is
        gs://elvos
    :param image_bucket: the bucket to store the images of the processed
        data in, currently this is gs://elvos-public
    :return:
    """
    for input_blob in data_bucket.list_blobs(prefix=f'airflow/npy'):
        input_filename = input_blob.name.split('/')[-1]
        arr = _download_arr(input_blob)

        logging.debug(f'processing numpy array')
        try:
            arr = process_arr(arr, crop_length, height_offset,
                              mip_thickness, pixel_value_range)
        except AssertionError:
            logging.debug(f'file {input_filename} could not be processed,'
                          f' has input shape {arr.shape}')
        else:
            _upload_npy(arr, data_name, input_filename, data_bucket)
            _upload_png(arr, data_name, input_filename, image_bucket)


def _download_arr(input_blob):
    logging.info(f'downloading numpy file: {input_blob.name}')
    input_stream = io.BytesIO()
    input_blob.download_to_file(input_stream)
    input_stream.seek(0)
    arr = np.load(input_stream)
    input_stream.close()
    return arr


def _upload_png(arr, data_name, input_filename, image_bucket):
    logging.debug(f'converting processed array into png')
    png_stream = io.BytesIO()
    # Standardize to [0, 1] otherwise it fails
    # TODO(luke): Consider standardizing by feature (RGB) as that is how
    # Keras does it
    standardized = (arr - arr.min()) / arr.max()
    plt.imsave(png_stream, standardized)
    png_stream.seek(0)
    png_filename = input_filename.replace('.npy', '.png')
    png_blob = image_bucket.blob(
        f'processed/{data_name}/arrays/{png_filename}')
    logging.info(f'uploading png file: {png_blob.name}')
    png_blob.upload_from_file(png_stream)
    png_stream.close()


def _upload_npy(arr, data_name, input_filename, data_bucket):
    arr_stream = io.BytesIO()
    np.save(arr_stream, arr)
    arr_stream.seek(0)
    arr_blob = data_bucket.blob(
        f'processed/{data_name}/arrays/{input_filename}')
    logging.info(f'uploading npy file: {arr_blob.name}')
    arr_blob.upload_from_file(arr_stream)


def process_arr(arr,
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


default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 24),
}

dag = DAG(dag_id='preprocess_web',
          description='Preprocesses data using a configuration passed by'
                      ' the web app.',
          default_args=default_args,
          schedule_interval=None,
          catchup=False)

preprocess_op = PreprocessOperator(task_id='preprocess',
                                   dag=dag)
