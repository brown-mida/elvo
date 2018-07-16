import keras
import numpy as np
import os
import pytest
import sklearn.preprocessing

import blueno.slack

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
                                     SLACK_TOKEN)
    assert r.status_code == 200


def test_full_multiclass_report_binary():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(1, activation='softmax'),
    ])

    X = np.random.rand(500, 224, 224, 3)
    y = np.random.randint(0, 2, size=(500, 1))

    print(blueno.slack.full_multiclass_report(model,
                                              X,
                                              y,
                                              classes=[0, 1]))


def test_full_multiclass_report_multiclass():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(3, activation='softmax'),
    ])

    X = np.random.rand(500, 224, 224, 3)
    y = np.random.randint(0, 3, size=(500,))
    y = sklearn.preprocessing.label_binarize(y, classes=[0, 1, 2])

    assert y.shape == (500, 3)

    print(blueno.slack.full_multiclass_report(model,
                                              X,
                                              y,
                                              classes=[0, 1, 2]))


def test_slack_upload_cm():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3)),
        keras.layers.Dense(3, activation='softmax'),
    ])

    X = np.random.rand(500, 224, 224, 3)
    y = np.random.randint(0, 3, size=(500,))
    y = sklearn.preprocessing.label_binarize(y, classes=[0, 1, 2])

    assert y.shape == (500, 3)

    report = blueno.slack.full_multiclass_report(model,
                                                 X,
                                                 y,
                                                 classes=[0, 1, 2])
    blueno.slack.upload_to_slack('/tmp/cm.png', report, SLACK_TOKEN)


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
    blueno.slack.save_misclassification_plots(X,
                                              y_valid_binary,
                                              y_pred_binary)
    blueno.slack.upload_to_slack('/tmp/false_positives.png',
                                 'false positives',
                                 SLACK_TOKEN)
    blueno.slack.upload_to_slack('/tmp/false_negatives.png',
                                 'false negatives',
                                 SLACK_TOKEN)
    blueno.slack.upload_to_slack('/tmp/true_positives.png',
                                 'true positives',
                                 SLACK_TOKEN)
    blueno.slack.upload_to_slack('/tmp/true_negatives.png',
                                 'true negatives',
                                 SLACK_TOKEN)


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
    blueno.slack.save_misclassification_plots(X,
                                              y_valid_binary,
                                              y_pred_binary,
                                              ids)
    blueno.slack.upload_to_slack('/tmp/false_positives.png',
                                 'false positives',
                                 SLACK_TOKEN)
    blueno.slack.upload_to_slack('/tmp/false_negatives.png',
                                 'false negatives',
                                 SLACK_TOKEN)
    blueno.slack.upload_to_slack('/tmp/true_positives.png',
                                 'true positives',
                                 SLACK_TOKEN)
    blueno.slack.upload_to_slack('/tmp/true_negatives.png',
                                 'true negatives',
                                 SLACK_TOKEN)
