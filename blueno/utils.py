"""Custom metrics, callbacks, and plots.
"""
import logging

import keras
import numpy as np
import sklearn.metrics
from keras import backend as K


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
        logging.info('val_auc: {}'.format(score))


class CustomReduceLR(keras.callbacks.ReduceLROnPlateau):
    """
    A LR reduction callback that will not lower the lr before
    epoch 25.
    """

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 25:
            super(CustomReduceLR, self).on_batch_end(epoch, logs)


def create_callbacks(x_train: np.ndarray, y_train: np.ndarray,
                     x_valid: np.ndarray, y_valid: np.ndarray,
                     early_stopping: bool = True,
                     reduce_lr: bool = False,
                     csv_file: str = None,
                     model_file: str = None,
                     normalize=True):
    """
    Instantiates a list of callbacks to be fed into the model.

    If csv_file is not None, a CSV logger is instantiated.
    If model_file is not None, a model checkpointer is instantiated.
    If early_stopping is true, an early stopping callback is created.

    The AUC callback is always created.

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param early_stopping: bool flag for whether to do early stopping
        on val_acc
    :param reduce_lr: bool flag for for whether or not to reduce the
        learning rate upon plateau
    :param csv_file: the filepath to save the CSV results to
    :param model_file: the filepath to save the models to
    :param normalize: whether or not to normalize the x data
    :return:
    """
    callbacks = []

    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_acc',
            verbose=1,
            patience=10
        ))

    if reduce_lr:
        callbacks.append(CustomReduceLR(
            monitor='val_acc',
            factor=0.1,
            verbose=1,
        ))

    if csv_file:
        callbacks.append(keras.callbacks.CSVLogger(csv_file, append=True))

    if model_file:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            model_file,
            monitor='val_acc',
            verbose=1,
            save_best_only=True
        ))

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

    callbacks.append(AucCallback(x_valid_standardized, y_valid))

    return callbacks
