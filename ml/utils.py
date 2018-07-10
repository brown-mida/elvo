"""Custom metrics, callbacks, and plots.
"""
import itertools
import logging

import keras
import matplotlib
import numpy as np
import requests
import sklearn.metrics
from keras import backend as K

import config

matplotlib.use('Agg')  # noqa: E402
from matplotlib import pyplot as plt


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def false_negatives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


class AucCallback(keras.callbacks.Callback):

    def __init__(self,
                 x_valid_standardized: np.ndarray,
                 y_valid: np.ndarray):
        super().__init__()
        self.x_valid_standardized = x_valid_standardized
        self.y_valid = y_valid

    def on_epoch_end(self, epoch: int, logs=None):
        y_pred = self.model.predict(self.x_valid_standardized)
        score = sklearn.metrics.roc_auc_score(self.y_valid, y_pred)
        print(f'\nval_auc: {score}')


def create_callbacks(x_train: np.ndarray, y_train: np.ndarray,
                     x_valid: np.ndarray, y_valid: np.ndarray, filename: str,
                     normalize=True):
    """
    Instantiates a list of callbacks:
    - AUC
    - Early stopping
    - TODO: model checkpoint

    :param normalize:
    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :return:
    """
    callbacks = []
    callbacks.append(keras.callbacks.CSVLogger(filename, append=True))
    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_acc',
                                                   patience=10))

    if normalize:
        x_mean = np.array([x_train[:, :, :, 0].mean(),
                           x_train[:, :, :, 1].mean(),
                           x_train[:, :, :, 2].mean()])
        x_std = np.array([x_train[:, :, :, 0].std(),
                          x_train[:, :, :, 1].std(),
                          x_train[:, :, :, 2].std()])
        x_valid_standardized = (x_valid - x_mean) / x_std
    else:
        x_valid_standardized = x_valid

    if y_valid.ndim == 1:
        # TODO: ROC for softmax
        callbacks.append(AucCallback(x_valid_standardized, y_valid))

    return callbacks


def save_history(history: keras.callbacks.History):
    loss_list = [s for s in history.history.keys() if
                 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if
                     'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if
                'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if
                    'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    epochs = range(1, len(history.history[loss_list[0]]) + 1)
    plt.figure()
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(
                     str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(
                     str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # TODO: Refactor so it's testable and no hard coded path
    plt.savefig('/tmp/loss.png')

    plt.figure()
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(
                     format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(
                     format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # TODO: Refactor so it's testable and no hard coded path
    plt.savefig('/tmp/acc.png')


def save_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('/tmp/cm.png')


def full_multiclass_report(model: keras.models.Model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    y_proba = model.predict(x, batch_size=batch_size)

    if y_proba.shape[-1] == 1:
        logging.debug('in multiclass_report, using >0.5 branch')
        y_pred = (y_proba > 0.5).astype('int32')
    else:
        y_pred = y_proba.argmax(axis=-1)

    assert y_pred.shape == y_true.shape, \
        f'y_pred.shape: {y_pred.shape} must equal y_true.shape: {y_true.shape}'

    comment = "Accuracy : " + str(
        sklearn.metrics.accuracy_score(y_true, y_pred))

    comment += '\n\n'

    comment += "Classification Report\n"
    comment += sklearn.metrics.classification_report(y_true, y_pred, digits=5)

    cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    comment += '\n'
    comment += str(cnf_matrix)
    save_confusion_matrix(cnf_matrix, classes=classes)
    return comment


def upload_to_slack(filename, comment):
    my_file = {
        'file': (filename, open(filename, 'rb'), 'png')
    }

    print(my_file)

    payload = {
        "filename": "history.png",
        "token": config.SLACK_TOKEN,
        'initial_comment': comment,
        "channels": ['#model-results'],
    }

    r = requests.post("https://slack.com/api/files.upload",
                      params=payload,
                      files=my_file)
    return r


def slack_report(x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_valid: np.ndarray,
                 y_valid: np.ndarray,
                 model: keras.models.Model,
                 history: keras.callbacks.History,
                 name: str,
                 params: dict):
    """
    Uploads a loss graph, accuacy, and confusion matrix plots in addition
    to useful data about the model to Slack.

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param model:
    :param history:
    :param name:
    :param params:
    :return:
    """
    save_history(history)
    upload_to_slack('/tmp/loss.png', f'{name}\n\nparams:\n{str(params)}')
    upload_to_slack('/tmp/acc.png', f'{name}\n\nparams:\n{str(params)}')

    x_mean = np.array([x_train[:, :, :, 0].mean(),
                       x_train[:, :, :, 1].mean(),
                       x_train[:, :, :, 2].mean()])
    x_std = np.array([x_train[:, :, :, 0].std(),
                      x_train[:, :, :, 1].std(),
                      x_train[:, :, :, 2].std()])
    x_valid_standardized = (x_valid - x_mean) / x_std

    if y_valid.ndim == 1:
        binary = True
    else:
        binary = False

    report = full_multiclass_report(model,
                                    x_valid_standardized,
                                    y_valid,
                                    [0, 1],
                                    batch_size=params['model']['batch_size'],
                                    binary=binary)
    upload_to_slack('/tmp/cm.png', report)
