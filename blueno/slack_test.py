import pathlib

import keras
import numpy as np
import os
import pytest
import sklearn.preprocessing

import blueno.slack

os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    from config_luke import SLACK_TOKEN
except ImportError:
    print('slack token missing')
    SLACK_TOKEN = ''


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses token only on gpu1708')
def test_upload_to_slack():
    with open('test_upload_to_slack.png', 'w') as f:
        f.write('hello!')
    r = blueno.slack.upload_to_slack('test_upload_to_slack.png',
                                     'testing',
                                     SLACK_TOKEN,
                                     channels=['#tests'])
    os.remove('test_upload_to_slack.png')
    assert r.status_code == 200


def test_full_multiclass_report_binary():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(1, activation='softmax'),
    ])

    X = np.random.rand(500, 224, 224, 3)
    y = np.random.randint(0, 2, size=(500, 1))

    cm_path = pathlib.Path('/tmp/test_cm.png')
    tp_path = pathlib.Path('/tmp/test_true_positives.png')
    fp_path = pathlib.Path('/tmp/test_false_positives.png')
    tn_path = pathlib.Path('/tmp/test_true_negatives.png')
    fn_path = pathlib.Path('/tmp/test_false_negatives.png')
    print(blueno.slack.full_multiclass_report(model,
                                              X,
                                              y,
                                              classes=[0, 1],
                                              cm_path=cm_path,
                                              tp_path=tp_path,
                                              fp_path=fp_path,
                                              tn_path=tn_path,
                                              fn_path=fn_path))


def test_full_multiclass_report_multiclass():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(3, activation='softmax'),
    ])

    X = np.random.rand(500, 224, 224, 3)
    y = np.random.randint(0, 3, size=(500,))
    y = sklearn.preprocessing.label_binarize(y, classes=[0, 1, 2])

    assert y.shape == (500, 3)

    cm_path = pathlib.Path('/tmp/test_cm.png')
    tp_path = pathlib.Path('/tmp/test_true_positives.png')
    fp_path = pathlib.Path('/tmp/test_false_positives.png')
    tn_path = pathlib.Path('/tmp/test_true_negatives.png')
    fn_path = pathlib.Path('/tmp/test_false_negatives.png')
    print(blueno.slack.full_multiclass_report(model,
                                              X,
                                              y,
                                              classes=[0, 1, 2],
                                              cm_path=cm_path,
                                              tp_path=tp_path,
                                              fp_path=fp_path,
                                              tn_path=tn_path,
                                              fn_path=fn_path))


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses token only on gpu1708')
def test_slack_upload_cm_two_classes():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(2, activation='softmax'),
    ])

    X = np.random.rand(500, 224, 224, 3)
    y = model.predict(X)

    assert y.shape == (500, 2)

    cm_path = pathlib.Path('/tmp/test_cm.png')
    tp_path = pathlib.Path('/tmp/test_true_positives.png')
    fp_path = pathlib.Path('/tmp/test_false_positives.png')
    tn_path = pathlib.Path('/tmp/test_true_negatives.png')
    fn_path = pathlib.Path('/tmp/test_false_negatives.png')

    report = blueno.slack.full_multiclass_report(model,
                                                 X,
                                                 y,
                                                 classes=[0, 1],
                                                 cm_path=cm_path,
                                                 tp_path=tp_path,
                                                 fp_path=fp_path,
                                                 tn_path=tn_path,
                                                 fn_path=fn_path)
    blueno.slack.upload_to_slack('/tmp/cm.png', report, SLACK_TOKEN,
                                 channels=['#tests'])


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses token only on gpu1708')
def test_slack_upload_cm_three_classes():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(3, activation='softmax'),
    ])

    X = np.random.rand(500, 224, 224, 3)
    y = np.random.randint(0, 3, size=(500,))
    y = sklearn.preprocessing.label_binarize(y, classes=[0, 1, 2])

    assert y.shape == (500, 3)

    cm_path = pathlib.Path('/tmp/test_cm.png')
    tp_path = pathlib.Path('/tmp/test_true_positives.png')
    fp_path = pathlib.Path('/tmp/test_false_positives.png')
    tn_path = pathlib.Path('/tmp/test_true_negatives.png')
    fn_path = pathlib.Path('/tmp/test_false_negatives.png')

    report = blueno.slack.full_multiclass_report(model,
                                                 X,
                                                 y,
                                                 classes=[0, 1, 2],
                                                 cm_path=cm_path,
                                                 tp_path=tp_path,
                                                 fp_path=fp_path,
                                                 tn_path=tn_path,
                                                 fn_path=fn_path)
    blueno.slack.upload_to_slack('/tmp/cm.png', report, SLACK_TOKEN,
                                 channels=['#tests'])


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses token only on gpu1708')
def test_save_misclassification_plots():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(3, activation='softmax'),
    ])
    X = np.random.rand(50, 224, 224, 3)
    y = np.random.randint(0, 3, size=(50,))
    y = sklearn.preprocessing.label_binarize(y, [0, 1, 2])
    y_pred = model.predict(X)

    y_valid = y.argmax(axis=1)
    y_valid_binary = y_valid > 0
    y_pred = y_pred.argmax(axis=1)
    y_pred_binary = y_pred > 0

    print('starting save')
    plot_dir = '/tmp'
    tp_path = pathlib.Path(plot_dir) / 'true_positives.png'
    fp_path = pathlib.Path(plot_dir) / 'false_positives.png'
    tn_path = pathlib.Path(plot_dir) / 'true_negatives.png'
    fn_path = pathlib.Path(plot_dir) / 'false_negatives.png'
    blueno.slack.save_misclassification_plots(X,
                                              y_valid_binary,
                                              y_pred_binary,
                                              tp_path=tp_path,
                                              fp_path=fp_path,
                                              tn_path=tn_path,
                                              fn_path=fn_path)
    blueno.slack.upload_to_slack('/tmp/false_positives.png',
                                 'false positives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])
    blueno.slack.upload_to_slack('/tmp/false_negatives.png',
                                 'false negatives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])
    blueno.slack.upload_to_slack('/tmp/true_positives.png',
                                 'true positives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])
    blueno.slack.upload_to_slack('/tmp/true_negatives.png',
                                 'true negatives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])


@pytest.mark.skipif(os.uname().nodename != 'gpu1708',
                    reason='Test uses token only on gpu1708')
def test_save_misclassification_plots_with_ids():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(3, activation='softmax'),
    ])
    X = np.random.rand(50, 224, 224, 3)
    y = np.random.randint(0, 3, size=(50,))
    ids = np.array([chr(i + 26) for i in range(50)])
    y = sklearn.preprocessing.label_binarize(y, [0, 1, 2])
    y_pred = model.predict(X)

    y_valid = y.argmax(axis=1)
    y_valid_binary = y_valid > 0
    y_pred = y_pred.argmax(axis=1)
    y_pred_binary = y_pred > 0

    print('starting save')
    tp_path = pathlib.Path('/tmp/test_true_positives.png')
    fp_path = pathlib.Path('/tmp/test_false_positives.png')
    tn_path = pathlib.Path('/tmp/test_true_negatives.png')
    fn_path = pathlib.Path('/tmp/test_false_negatives.png')
    blueno.slack.save_misclassification_plots(X,
                                              y_valid_binary,
                                              y_pred_binary,
                                              tp_path=tp_path,
                                              fp_path=fp_path,
                                              tn_path=tn_path,
                                              fn_path=fn_path,
                                              id_valid=ids)
    blueno.slack.upload_to_slack('/tmp/false_positives.png',
                                 'false positives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])
    blueno.slack.upload_to_slack('/tmp/false_negatives.png',
                                 'false negatives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])
    blueno.slack.upload_to_slack('/tmp/true_positives.png',
                                 'true positives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])
    blueno.slack.upload_to_slack('/tmp/true_negatives.png',
                                 'true negatives',
                                 SLACK_TOKEN,
                                 channels=['#tests'])
