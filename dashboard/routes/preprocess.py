import datetime
import logging
from multiprocessing import Process

import flask
import pymongo

from utils import gcs, preprocess, dbx
from utils.mongodb import get_db

app_preprocess = flask.Blueprint('app_preprocess', __name__)


@app_preprocess.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    data = flask.request.form
    files = flask.request.files

    if 'user' not in data:
        return flask.json.jsonify({'error': 'User not specified'})

    if 'file' not in files:
        logging.debug("File not found")
        return flask.json.jsonify({'error': 'No file'})

    files = files.getlist('file')
    logging.debug(files)
    logging.debug(str(datetime.datetime.now()))

    client = pymongo.MongoClient(
        "mongodb://bluenoml:elvoanalysis@104.196.51.205/elvo"
    )
    datasets = client.elvo.datasets
    client = gcs.authenticate()
    bucket = client.get_bucket('blueno-ml-files')

    # TODO: Make multiprocessing work; it doesn't work because multiple
    # consecutive requests to GCS seems to throw an error.
    current_date = str(datetime.datetime.now())
    for file in files:

        # Save user-file relationship in MongoDB
        dataset = {
            "user": data['user'],
            "name": file.filename,
            "date": current_date,
            "mip": False,
            "shape": "None",
            "dataset": 'default',
            "gcs_url": '{}/default/{}.npy'.format(data['user'], file.filename),
            "status": "running",
            "message": "Please wait for the file to finish loading."
        }
        datasets.replace_one({'name': file.filename, 'dataset': 'default'},
                             dataset, upsert=True)

    p = Process(target=__process_file,
                args=(files, data['user'], current_date, bucket))
    p.start()
    return flask.json.jsonify({'status': 'success'})


def __process_file(files, user, date, bucket):
    client = pymongo.MongoClient(
        "mongodb://bluenoml:elvoanalysis@104.196.51.205/elvo"
    )
    db = client.elvo.datasets
    for file in files:
        # try:
        npy = preprocess.process_cab_from_file(file, file.filename,
                                               '../tmp/cab_files/')
        preprocess.generate_images(npy, user, 'default', file.filename,
                                   bucket, '../tmp/')
        gcs.upload_npy_to_gcs(npy, file.filename, user, 'default', bucket)
        dataset = {
            "user": user,
            "name": file.filename,
            "date": date,
            "mip": False,
            "shape": npy.shape,
            "dataset": 'default',
            "gcs_url": '{}/default/{}.npy'.format(user, file.filename),
            "status": "loaded",
            "message": "Successfully loaded."
        }
        db.replace_one({'name': file.filename, 'dataset': 'default'},
                       dataset)
        # except Exception as e:
        #     dataset = {
        #         "user": user,
        #         "name": file.filename,
        #         "date": date,
        #         "mip": False,
        #         "shape": "None",
        #         "dataset": 'default',
        #         "gcs_url": '{}/default/{}.npy'.format(user, file.filename),
        #         "status": "failed",
        #         "message": "Failed: {}".format(e)
        #     }
        #     db.replace_one({'name': file.filename, 'dataset': 'default'},
        #                    dataset)


@app_preprocess.route('/upload-dropbox-dataset', methods=['POST'])
def upload_dataset_from_dropbox():
    data = flask.request.get_json()

    if 'user' not in data:
        return flask.json.jsonify({'error': 'User not specified'})

    if 'path' not in data:
        return flask.json.jsonify({'error': 'Path not specified'})

    if 'token' not in data:
        return flask.json.jsonify({'error': 'Token not specified'})

    db = get_db()
    dbx_client = dbx.authenticate(data['token'])

    try:
        dbx_client.users_get_current_account()
    except Exception:
        return flask.json.jsonify({'error': 'Invalid token'})

    try:
        files = dbx.get_cab_from_folder(dbx_client, data['path'])
    except Exception:
        return flask.json.jsonify({'error': 'Invalid path'})

    current_date = str(datetime.datetime.now())
    for file in files:
        # Save user-file relationship in MongoDB
        dataset = {
            "user": data['user'],
            "name": file,
            "date": current_date,
            "mip": False,
            "shape": "None",
            "dataset": 'default',
            "gcs_url": '{}/default/{}.npy'.format(data['user'], file),
            "status": "running",
            "message": "Please wait for the file to finish loading."
        }
        db.replace_one({'name': file, 'dataset': 'default'},
                       dataset, upsert=True)

    p = Process(target=__process_dbx_file,
                args=(files, data['user'], current_date,
                      data['path'], data['token']))
    p.start()
    return flask.json.jsonify({'status': 'success'})


def __process_dbx_file(files, user, date, path, token):
    db = get_db()
    dbx_client = dbx.authenticate(token)
    client = gcs.authenticate()
    bucket = client.get_bucket('blueno-ml-files')

    for file in files:
        try:
            dbx.download_from_dropbox(dbx_client, '../tmp/cab_files/',
                                      path, file)
            npy = preprocess.process_cab(file, '../tmp/cab_files/')
            preprocess.generate_images(npy, user, 'default', file,
                                       bucket, '../tmp/')
            gcs.upload_npy_to_gcs(npy, file, user, 'default', bucket)
            dataset = {
                "user": user,
                "name": file,
                "date": date,
                "mip": False,
                "shape": npy.shape,
                "dataset": 'default',
                "gcs_url": '{}/default/{}.npy'.format(user, file),
                "status": "loaded",
                "message": "Successfully loaded."
            }
            db.replace_one({'name': file, 'dataset': 'default'},
                           dataset)
        except Exception as e:
            dataset = {
                "user": user,
                "name": file,
                "date": date,
                "mip": False,
                "shape": "None",
                "dataset": 'default',
                "gcs_url": '{}/default/{}.npy'.format(user, file),
                "status": "failed",
                "message": "Failed: {}".format(e)
            }
            db.replace_one({'name': file, 'dataset': 'default'},
                           dataset)


@app_preprocess.route('/get-dataset', methods=['GET'])
def get_dataset():
    client = pymongo.MongoClient(
        "mongodb://bluenoml:elvoanalysis@104.196.51.205/elvo"
    )
    datasets = client.elvo.datasets
    user = flask.request.args.get('user')
    dataset = flask.request.args.get('dataset')
    results = datasets.find({'user': user, 'dataset': dataset})
    data = []
    for doc in results:
        data.append({'name': doc['name'],
                     'date': doc['date'],
                     'dataset': doc['dataset'],
                     'mip': doc['mip'],
                     'status': doc['status'],
                     'shape': doc['shape'],
                     'message': doc['message']})
    return flask.json.jsonify({'data': data})


@app_preprocess.route('/get-dataset-image', methods=['GET'])
def get_dataset_image():
    user = flask.request.args.get('user')
    dataset = flask.request.args.get('dataset')
    data_type = flask.request.args.get('type')
    data_name = flask.request.args.get('name')
    logging.info(user)

    client = gcs.authenticate()
    bucket = client.get_bucket('blueno-ml-files')
    image = gcs.download_image(user, dataset, data_type, data_name, bucket)
    return flask.send_file(image, mimetype='image/jpg')


@app_preprocess.route('/get-datasets-from-user', methods=['GET'])
def get_datasets_from_user():
    user = flask.request.args.get('user')
    db = get_db()
    results = db.find({'user': user}).distinct('dataset')
    return flask.json.jsonify({'data': results})


@app_preprocess.route('/new-preprocessing-job', methods=['POST'])
def make_preprocessing_job():
    data = flask.request.get_json()

    # Get list of images to preprocess
    db = get_db()
    image_list = []
    results = db.find({'user': data['user'], 'dataset': 'default'})
    for doc in results:
        image_list.append({'name': doc['name'], 'gcs_url': doc['gcs_url']})
    logging.info(image_list)
    logging.info("A")

    # Save new file relationship in MongoDB
    current_date = str(datetime.datetime.now())
    for image in image_list:
        dataset = {
            "user": data['user'],
            "name": image['name'],
            "mip": data['mip'],
            "shape": "None",
            "date": current_date,
            "dataset": data['name'],
            "gcs_url": '{}/{}/{}.npy'.format(data['user'], data['name'],
                                             image['name']),
            "status": "running",
            "message": "Currently processing this file."
        }
        db.replace_one({'name': image['name'], 'dataset': data['name']},
                       dataset, upsert=True)

    p = Process(target=__processing_job_helper,
                args=(image_list, data, current_date))
    p.start()
    return flask.json.jsonify({'status': 'success'})


def __processing_job_helper(image_list, data, current_date):
    client = gcs.authenticate()
    bucket = client.get_bucket('blueno-ml-files')
    db = get_db()
    for image in image_list:
        try:
            logging.info(data)
            arr = gcs.download_array(image['gcs_url'], bucket)
            arr = preprocess.transform_array(arr, data)
            if data['mip']:
                preprocess.generate_mip_images(
                    arr, data['user'], data['name'],
                    image['name'], bucket, '../tmp/')
            else:
                preprocess.generate_images(arr, data['user'], data['name'],
                                           image['name'], bucket, '../tmp/')
            gcs.upload_npy_to_gcs(arr, image['name'], data['user'],
                                  data['name'], bucket)
            dataset = {
                "user": data['user'],
                "name": image['name'],
                "mip": data['mip'],
                "shape": arr.shape,
                "date": current_date,
                "dataset": data['name'],
                "gcs_url": '{}/{}/{}.npy'.format(data['user'], data['name'],
                                                 image['name']),
                "status": "loaded",
                "message": "Successfully loaded."
            }
            db.replace_one({'name': image['name'], 'dataset': data['name']},
                           dataset)
            logging.info(arr.shape)
        except Exception as e:
            dataset = {
                "user": data['user'],
                "name": image['name'],
                "mip": data['mip'],
                "shape": "None",
                "date": current_date,
                "dataset": data['name'],
                "gcs_url": '{}/{}/{}.npy'.format(data['user'], data['name'],
                                                 image['name']),
                "status": "failed",
                "message": "Failed: {}".format(e)
            }
            db.replace_one({'name': image['name'], 'dataset': data['name']},
                           dataset)
