import datetime
import logging

import flask
import pymongo
from multiprocessing import Process

from utils import gcs, preprocess

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
    for file in files:
        current_date = str(datetime.datetime.now())
        file_id = ('{}.{}'.format(file.filename, current_date))

        # Upload file
        process = Process(target=process_file,
                          args=(file, file_id, data['user'],
                                current_date, bucket))
        process.start()

        # Save user-file relationship in MongoDB
        dataset = {
            "user": data['user'],
            "id": file_id,
            "name": file.filename,
            "date": current_date,
            "gcs_url": '{}/default/{}.npy'.format(data['user'], file_id),
            "status": "running",
            "message": "Please wait for the file to finish loading."
        }
        tmp = datasets.insert_one(dataset).inserted_id
        logging.debug(tmp)

    return flask.json.jsonify({'status': 'success'})


@app_preprocess.route('/get-dataset', methods=['GET'])
def get_dataset():
    client = pymongo.MongoClient(
        "mongodb://bluenoml:elvoanalysis@104.196.51.205/elvo"
    )
    datasets = client.elvo.datasets
    user = flask.request.args.get('user')
    results = datasets.find({'user': user})
    data = []
    for doc in results:
        data.append({'name': doc['name'],
                     'id': doc['id'],
                     'date': doc['date'],
                     'status': doc['status'],
                     'message': doc['message']})
    return flask.json.jsonify({'data': data})


def process_file(file, file_id, user, date, bucket):
    client = pymongo.MongoClient(
        "mongodb://bluenoml:elvoanalysis@104.196.51.205/elvo"
    )
    db = client.elvo.datasets
    try:
        npy = preprocess.process_cab(file, file.filename, '../tmp/cab_files/')
        preprocess.generate_images(npy, user, 'default', file_id,
                                   bucket, '../tmp/')
        gcs.upload_npy_to_gcs(npy, file_id, user, 'default', bucket)
        dataset = {
            "user": user,
            "id": file_id,
            "name": file.filename,
            "date": date,
            "gcs_url": '{}/default/{}.npy'.format(user, file_id),
            "status": "loaded",
            "message": "Successfully loaded."
        }
        db.replace_one({'id': file_id}, dataset)
    except Exception as e:
        dataset = {
            "user": user,
            "id": file_id,
            "name": file.filename,
            "date": date,
            "gcs_url": '{}/default/{}.npy'.format(user, file_id),
            "status": "failed",
            "message": "Failed: {}".format(e)
        }
        db.replace_one({'id': file_id}, dataset)

