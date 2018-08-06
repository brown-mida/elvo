import ast
import pathlib
import typing
from collections.__init__ import namedtuple

import elasticsearch_dsl
import pandas as pd
import re
from pandas.errors import EmptyDataError

TRAINING_JOBS = 'training_jobs'
VALIDATION_JOBS = 'validation_jobs'
JOB_INDEX = elasticsearch_dsl.Index(TRAINING_JOBS)
VALIDATION_JOB_INDEX = elasticsearch_dsl.Index(VALIDATION_JOBS)

Metrics = namedtuple('Metrics', ['epochs',
                                 'train_acc',
                                 'final_val_acc',
                                 'best_val_acc',
                                 'final_val_loss',
                                 'best_val_loss',
                                 'final_val_sensitivity',
                                 'best_val_sensitivity',
                                 'final_val_specificity',
                                 'best_val_specificity'])


class TrainingJob(elasticsearch_dsl.Document):
    id = elasticsearch_dsl.Integer()
    schema_version = elasticsearch_dsl.Integer()
    job_name = elasticsearch_dsl.Keyword()
    author = elasticsearch_dsl.Keyword()
    created_at = elasticsearch_dsl.Date()
    ended_at = elasticsearch_dsl.Date()
    params = elasticsearch_dsl.Text()
    raw_log = elasticsearch_dsl.Text()
    model_url = elasticsearch_dsl.Text()

    # Metrics
    epochs = elasticsearch_dsl.Integer()
    train_acc = elasticsearch_dsl.Float()
    final_val_acc = elasticsearch_dsl.Float()
    best_val_acc = elasticsearch_dsl.Float()
    final_val_loss = elasticsearch_dsl.Float()
    best_val_loss = elasticsearch_dsl.Float()
    final_val_sensitivity = elasticsearch_dsl.Float()
    best_val_sensitivity = elasticsearch_dsl.Float()
    final_val_specificity = elasticsearch_dsl.Float()
    best_val_specificity = elasticsearch_dsl.Float()
    final_val_auc = elasticsearch_dsl.Float()
    best_val_auc = elasticsearch_dsl.Float()

    # Params
    batch_size = elasticsearch_dsl.Integer()
    val_split = elasticsearch_dsl.Float()
    seed = elasticsearch_dsl.Integer()

    rotation_range = elasticsearch_dsl.Float()
    width_shift_range = elasticsearch_dsl.Float()
    height_shift_range: float = elasticsearch_dsl.Float()
    shear_range = elasticsearch_dsl.Float()
    zoom_range = elasticsearch_dsl.Keyword()
    horizontal_flip = elasticsearch_dsl.Boolean()
    vertical_flip = elasticsearch_dsl.Boolean()

    dropout_rate1 = elasticsearch_dsl.Float()
    dropout_rate2 = elasticsearch_dsl.Float()

    data_dir = elasticsearch_dsl.Keyword()
    gcs_url = elasticsearch_dsl.Keyword()

    mip_thickness = elasticsearch_dsl.Integer()
    height_offset = elasticsearch_dsl.Integer()
    pixel_value_range = elasticsearch_dsl.Keyword()

    # We need to keep a list of params for the parser because
    # we can't use traditional approaches to get the class attrs
    params_to_parse = ['batch_size',
                       'val_split',
                       'seed',
                       'rotation_range',
                       'width_shift_range',
                       'height_shift_range',
                       'shear_range',
                       'zoom_range',
                       'horizontal_flip',
                       'vertical_flip',
                       'dropout_rate1',
                       'dropout_rate2',
                       'data_dir',
                       'gcs_url',
                       'mip_thickness',
                       'height_offset',
                       'pixel_value_range']

    class Index:
        name = TRAINING_JOBS


class ValidationJob(elasticsearch_dsl.Document):
    """
    Object for validation data.
    TODO: Can this be merged with TrainingJob, with a common
        parent object?
    """
    id = elasticsearch_dsl.Integer()
    schema_version = elasticsearch_dsl.Integer()
    job_name = elasticsearch_dsl.Keyword()
    author = elasticsearch_dsl.Keyword()
    created_at = elasticsearch_dsl.Date()
    params = elasticsearch_dsl.Text()
    raw_log = elasticsearch_dsl.Text()

    # Metrics
    purported_acc = elasticsearch_dsl.Float()
    purported_loss = elasticsearch_dsl.Float()
    purported_sensitivity = elasticsearch_dsl.Float()

    avg_test_acc = elasticsearch_dsl.Float()
    avg_test_loss = elasticsearch_dsl.Float()
    avg_test_sensitivity = elasticsearch_dsl.Float()
    avg_test_specificity = elasticsearch_dsl.Float()
    avg_test_true_pos = elasticsearch_dsl.Float()
    avg_test_false_neg = elasticsearch_dsl.Float()
    avg_test_auc = elasticsearch_dsl.Float()

    best_test_acc = elasticsearch_dsl.Float()
    best_test_loss = elasticsearch_dsl.Float()
    best_test_sensitivity = elasticsearch_dsl.Float()
    best_test_specificity = elasticsearch_dsl.Float()
    best_test_true_pos = elasticsearch_dsl.Float()
    best_test_false_neg = elasticsearch_dsl.Float()
    best_test_auc = elasticsearch_dsl.Float()
    best_end_val_acc = elasticsearch_dsl.Float()
    best_end_val_loss = elasticsearch_dsl.Float()
    best_max_val_acc = elasticsearch_dsl.Float()
    best_max_val_loss = elasticsearch_dsl.Float()

    # Params
    batch_size = elasticsearch_dsl.Integer()
    val_split = elasticsearch_dsl.Float()
    seed = elasticsearch_dsl.Integer()

    rotation_range = elasticsearch_dsl.Float()
    width_shift_range = elasticsearch_dsl.Float()
    height_shift_range = elasticsearch_dsl.Float()
    shear_range = elasticsearch_dsl.Float()
    zoom_range = elasticsearch_dsl.Keyword()
    horizontal_flip = elasticsearch_dsl.Boolean()
    vertical_flip = elasticsearch_dsl.Boolean()

    dropout_rate1 = elasticsearch_dsl.Float()
    dropout_rate2 = elasticsearch_dsl.Float()

    data_dir = elasticsearch_dsl.Keyword()
    gcs_url = elasticsearch_dsl.Keyword()

    mip_thickness = elasticsearch_dsl.Integer()
    height_offset = elasticsearch_dsl.Integer()
    pixel_value_range = elasticsearch_dsl.Keyword()

    # We need to keep a list of params for the parser because
    # we can't use traditional approaches to get the class attrs
    params_to_parse = ['batch_size',
                       'val_split',
                       'seed',
                       'rotation_range',
                       'width_shift_range',
                       'height_shift_range',
                       'shear_range',
                       'zoom_range',
                       'horizontal_flip',
                       'vertical_flip',
                       'dropout_rate1',
                       'dropout_rate2',
                       'data_dir',
                       'gcs_url',
                       'mip_thickness',
                       'height_offset',
                       'pixel_value_range']

    class Index:
        name = VALIDATION_JOBS


def insert_or_replace_filepaths(log_file: pathlib.Path,
                                csv_file: typing.Optional[pathlib.Path],
                                gpu1708=False,
                                alias='default'):
    training_job = construct_job_from_filepaths(log_file,
                                                csv_file,
                                                gpu1708)
    if training_job is None:
        print('training job is none, not inserting')
        return
    insert_or_replace(training_job, alias=alias)


def insert_or_ignore_filepaths(log_file: pathlib.Path,
                               csv_file: typing.Optional[pathlib.Path],
                               gpu1708=False,
                               alias='default'):
    """
    Parses matching log file and csv and uploads the file up to the
    Elasticsearch index, if it doesn't exist.

    Note that the parsing is very brittle, so important logs
    should be documented in bluenot.py

    :param log_file:
    :param csv_file:
    :param gpu1708:
    :return:
    """
    training_job = construct_job_from_filepaths(log_file,
                                                csv_file,
                                                gpu1708)
    if training_job is None:
        print('training job is none, not inserting')
        return
    insert_or_ignore(training_job, alias=alias)


def insert_or_replace(training_job: TrainingJob, alias='default'):
    """Inserts the training job into the TrainingJob index, replacing
    the old match, if one exists.

    Raises a ValueError if multiple matches exist.
    """
    if 'slack' not in training_job.raw_log:
        print('job is incomplete, returning')
        return

    search: elasticsearch_dsl.Search = JOB_INDEX.search(using=alias) \
        .query('match', job_name=training_job.job_name) \
        .query('match', created_at=training_job.created_at)

    if len(search.execute()) > 1:
        raise ValueError(f'Found two matches to {training_job.job_name}'
                         f'created at {training_job.created_at}')
    elif len(search.execute()) == 1:
        print('replacing result {} created at'.format(training_job.job_name,
                                                      training_job.created_at))
    else:
        print('no match found for {} created at'.format(
            training_job.job_name,
            training_job.created_at))

    search.delete()
    training_job.save(using=alias)


def insert_or_ignore(job: elasticsearch_dsl.Document, alias='default',
                     index=JOB_INDEX):
    """Inserts the training job into the elasticsearch index
    if no job with the same name and creation timestamp exists.
    """
    if index == JOB_INDEX and 'slack' not in job.raw_log:
        print('job is incomplete, returning')
        return

    matches = index.search() \
        .query('match', job_name=job.job_name) \
        .query('match', created_at=job.created_at) \
        .count()

    if matches == 0:
        job.save(using=alias)
    else:
        print('job {} created at {} exists'.format(
            job.job_name, job.created_at))


def construct_job_from_filepaths(
        log_file: pathlib.Path,
        csv_file: typing.Optional[pathlib.Path],
        gpu1708=False) -> typing.Optional[TrainingJob]:
    """
    Creates a TrainingJob instance from the given files.

    Returns None if a job cannot be created.

    :param log_file:
    :param csv_file:
    :param gpu1708:
    :param alias:
    :return:
    """
    filename = str(log_file.name)

    job_name, created_at = _parse_filename(filename)
    params = _extract_params(log_file)
    author = _extract_author(log_file)
    raw_log = open(log_file).read()
    ended_at = _extract_ended_at(log_file)
    model_url = _extract_model_url(log_file)
    final_val_auc = _extract_auc(log_file)
    best_val_auc = _extract_best_auc(log_file)

    if author is None and gpu1708:
        author = _fill_author_gpu1708(created_at, job_name)

    if params:
        params_dict = _parse_params_str(params)
    else:
        params_dict = None

    try:
        metrics = _extract_metrics(csv_file)
    except (ValueError, EmptyDataError):
        print('metrics file {} is empty'.format(csv_file))
        return None

    training_job = construct_job(job_name,
                                 created_at,
                                 params,
                                 raw_log,
                                 metrics,
                                 str(csv_file.name),
                                 author=author,
                                 ended_at=ended_at,
                                 model_url=model_url,
                                 final_val_auc=final_val_auc,
                                 best_val_auc=best_val_auc,
                                 params_dict=params_dict)
    return training_job


def construct_job(job_name,
                  created_at,
                  params: str,
                  raw_log,
                  metrics: Metrics,
                  metrics_filename,
                  author=None,
                  ended_at=None,
                  model_url=None,
                  final_val_auc=None,
                  best_val_auc=None,
                  params_dict=None) -> TrainingJob:
    """
    Constructs a training job object from the given parameters.

    Note that these parameters are experimental.
    :param job_name:
    :param created_at:
    :param params: a string of bluenot config params
    :param raw_log:
    :param metrics:
    :param metrics_filename:
    :param author:
    :param ended_at:
    :param model_url:
    :param final_val_auc:
    :return:
    """
    training_job = TrainingJob(schema_version=1,
                               job_name=job_name,
                               author=author,
                               created_at=created_at,
                               ended_at=ended_at,
                               params=params,
                               raw_log=raw_log,
                               model_url=model_url,
                               final_val_auc=final_val_auc,
                               best_val_auc=best_val_auc)

    if params_dict:
        for key, val in params_dict.items():
            if key is 'data_dir' and val.endswith('/'):  # standardize dirpaths
                val = val[:-1]
            training_job.__setattr__(key, val)

    if (job_name, created_at) == _parse_filename(metrics_filename):
        print('found matching CSV file, setting metrics')
        training_job.epochs = metrics.epochs
        training_job.train_acc = metrics.train_acc
        training_job.final_val_acc = metrics.final_val_acc
        training_job.best_val_acc = metrics.best_val_acc
        training_job.final_val_loss = metrics.final_val_loss
        training_job.best_val_loss = metrics.best_val_loss
        training_job.final_val_sensitivity = \
            metrics.final_val_sensitivity
        training_job.best_val_sensitivity = \
            metrics.best_val_sensitivity
        training_job.final_val_specificity = metrics.final_val_specificity
        training_job.best_val_specificity = metrics.best_val_specificity

    return training_job


def _parse_filename(filename: str) -> typing.Tuple[str, str]:
    """Parse a CSV or log file.
    """
    date_start_idx = filename.find('2018')
    job_name = filename[:date_start_idx - 1]
    created_at = filename[date_start_idx:-4]
    return job_name, created_at


def _extract_params(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if line.endswith('using params:\n'):
                second_line = f.readline()
                return second_line
        return None


def _extract_author(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if 'INFO - author:' in line:
                author = line.rstrip('\n').split(':')[-1].strip()
                return author
    return None


def _extract_ended_at(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if 'INFO - end time:' in line:
                return line.split(' ')[-1].rstrip('\n')
    return None


def _extract_model_url(log_path: pathlib.Path) -> typing.Optional[str]:
    with open(log_path) as f:
        for line in f:
            if 'uploading model' in line:
                return line.split(' ')[-1].rstrip('\n')
    return None


def _extract_auc(log_path: pathlib.Path) -> typing.Optional[float]:
    with open(log_path) as f:
        pattern = re.compile(r'initial_comment=Accuracy%3A\+(.+)'
                             r'AUC%3A\+(.+?)%0')
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(2))
    return None


def _extract_best_auc(log_path: pathlib.Path) -> typing.Optional[float]:
    aucs = []

    with open(log_path) as f:
        for line in f:
            if 'INFO - val_auc:' in line:
                try:
                    auc = float(line.split(' ')[-1].rstrip('\n'))
                    aucs.append(auc)
                except ValueError:
                    print('matching line could not be parsed: {}'.format(line))

    if len(aucs) == 0:
        return None

    return max(aucs)


def _parse_params_str(params_str: str) -> typing.Dict[str, typing.Any]:
    """Parses the param string outputs that most logs contain.

    This code is very rigid, and will likely break.
    """
    param_dict = {}
    for param in TrainingJob.params_to_parse:
        if param in ('zoom_range', 'pixel_value_range'):
            float_pattern = '[0-9]*\.?[0-9]+'
            pattern = r'{}=(\({}, {}\))[,)]'.format(
                param,
                float_pattern,
                float_pattern,
            )
            print(params_str)
            match = re.search(pattern, params_str)
            if match:
                param_dict[param] = match.group(1)
        else:
            patterns = [r'{}=(.*?)[,)]'.format(param),
                        r"'{}'".format(param) + r": (.*?)[,}]"]
            for pattern in patterns:
                match = re.search(pattern, params_str)
                if match:
                    value_str = match.group(1)
                    value = ast.literal_eval(value_str)
                    param_dict[param] = value
    print('parsed params:', param_dict)
    return param_dict


def _extract_metrics(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    return Metrics(epochs=df['epoch'].max(),
                   train_acc=df['acc'].iloc[-1],
                   final_val_acc=df['val_acc'].iloc[-1],
                   best_val_acc=df['val_acc'].max(),
                   final_val_loss=df['val_loss'].iloc[-1],
                   best_val_loss=df['val_loss'].min(),
                   final_val_sensitivity=df['val_sensitivity'].iloc[-1],
                   best_val_sensitivity=df['val_sensitivity'].max(),
                   final_val_specificity=df['val_specificity'].iloc[-1],
                   best_val_specificity=df['val_specificity'].max())


def _fill_author_gpu1708(created_at, job_name):
    if created_at > '2018-07-11' \
            or 'processed_' in job_name \
            or 'processed-no-vert' in job_name:
        author = 'sumera'
    else:
        author = 'luke'
    return author


def create_new_connection(address, index='training_jobs'):
    """
    Creates a new connection to elasticsearch.
    """
    elasticsearch_dsl.connections.create_connection(
        hosts=[address]
    )
    return elasticsearch_dsl.Index(index)


def search_top_models(address, lower=0.8, upper=0.923):
    """
    Search training models from Kibana.

    :params address: Address to connect to
    :params lower: Lower bound validation accuracy to search
    :params upper: Upper bound validation accuracy to search
    :returns: A ElasticSearch response object with bounded validation
        accuracy. Use a for loop to iterate through them.
    """
    index = create_new_connection(address)

    matches = index.search()
    matches = matches.filter('range',
                             best_val_acc={'gte': lower, 'lte': upper})
    count = matches.count()
    response = matches[0:count].execute()
    return response


def filter_top_models(address, models):
    """
    Take training params obtained from search_top_models, and cull
    params that have already been used and uploaded to the
    validation_jobs index in Kibana.

    :params address: Address to connect to
    :params models: Params list
    """
    index = create_new_connection(address, index='validation_jobs')

    matches = index.search()
    count = matches.count()
    response = matches[0:count].execute()

    result = []
    for model in models:
        flagged = False
        for r in response:
            if model.created_at == r.created_at:
                flagged = True
        if not flagged:
            result.append(model)
    return result


def get_validation_job_from_log(log_path):
    """
    Creates a ValidationJob from the log.
    """
    with open(log_path) as f:
        lines = f.read().splitlines()
        params = lines[1]
        job_name = lines[2].split(' ')[-1]
        author = lines[3].split(' ')[-1]
        job_date = lines[4].split(' ')[-1]
        purported_acc = lines[5].split(' ')[-1]
        purported_loss = lines[6].split(' ')[-1]
        purported_sensitivity = lines[7].split(' ')[-1]

        # Get the line indexes which display results.
        result_lines = []
        for i, line in enumerate(lines):
            if 'Results' in line:
                result_lines.append(i)

        # Get the final (averaged) metrics.
        avg_results = []
        for i in range(1, 8):
            avg_results.append(lines[result_lines[-1] + i].split(' ')[-1])

        # Get the best metrics from the n iterations.
        best_results = [[] for i in range(11)]
        for i in result_lines[:-1]:
            for j in range(11):
                best_results[j].append(lines[i + j + 2].split(' ')[-1])
        best_results = [max(li) for li in best_results]

        # Get params
        params_dict = _parse_params_str(params)

        job = ValidationJob(
            schema_version=1,
            job_name=job_name,
            author=author,
            created_at=job_date,
            params=params.split('INFO - ')[-1],
            raw_log='\n'.join(lines),
            purported_acc=purported_acc,
            purported_loss=purported_loss,
            purported_sensitivity=purported_sensitivity,
            avg_test_loss=avg_results[0],
            avg_test_acc=avg_results[1],
            avg_test_sensitivity=avg_results[2],
            avg_test_specificity=avg_results[3],
            avg_test_true_pos=avg_results[4],
            avg_test_false_neg=avg_results[5],
            avg_test_auc=avg_results[6],
            best_test_acc=best_results[0],
            best_test_loss=best_results[1],
            best_test_sensitivity=best_results[2],
            best_test_specificity=best_results[3],
            best_test_true_pos=best_results[4],
            best_test_false_neg=best_results[5],
            best_test_auc=best_results[6],
            best_end_val_acc=best_results[7],
            best_end_val_loss=best_results[8],
            best_max_val_acc=best_results[9],
            best_max_val_loss=best_results[10])

        if params_dict:
            for key, val in params_dict.items():
                if key is 'data_dir' and val.endswith('/'):
                    val = val[:-1]
                job.__setattr__(key, val)

        return job
