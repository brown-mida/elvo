"""
Connection logic with Google Cloud Storage.
"""

from google.cloud import storage

from blueno.reporting import JOB_INDEX


def fetch_model(service_account_path=None, save_path=None, **kwargs):
    """
    Downloads the relevant model from Google Cloud, given specific
    parameters of the model.

    e.g. fetch_model(_id='A8U2pGQBLx2QGQijrVch') will download the
        model with the specific id.

    :param service_account_path: The path of the GCS service account JSON.
    :param save_path: The path to save the model.
    :param kwargs: The series of parameters to specify the model.
    :return:
    :raise ValueError: If the parameter query returns no result, or
        returns more than 1 result.
    """
    if service_account_path is None:
        service_account_path = '../credentials/client_secret.json'

    if save_path is None:
        save_path = '../tmp/downloaded_models'

    params = kwargs
    matches = JOB_INDEX.search()
    for param in params.keys():
        d = {param: params[param]}
        matches = matches.query('match', **d)
    response = matches.execute()

    if len(response) > 1:
        raise ValueError(('Query not specific enough. '
                          'Found {} results'.format(len(response))))
    elif len(response) == 0:
        raise ValueError('Found 0 results.')

    result = list(response)[0]
    result_str = '{}-{}'.format(result.job_name, result.created_at)

    gcs_client = storage.Client.from_service_account_json(
        service_account_path
    )
    bucket = gcs_client.get_bucket('elvos')
    blob = storage.Blob('models/{}'.format(result_str), bucket)
    blob.download_to_filename(
        '{}/{}.hdf5'.format(save_path, result_str)
    )
