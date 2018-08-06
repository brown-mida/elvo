import itertools
import pathlib
import typing

import keras
import matplotlib
import numpy as np
import os
import pandas as pd
import requests
import sklearn.metrics

matplotlib.use('Agg')  # noqa: E402
from matplotlib import pyplot as plt


# TODO(luke): Split this into a slack and a plotting module.


def slack_report(x_train: np.ndarray,
                 x_valid: np.ndarray,
                 y_valid: np.ndarray,
                 model: keras.models.Model,
                 history: keras.callbacks.History,
                 name: str,
                 params: typing.Any,
                 token: str,
                 id_valid: np.ndarray = None,
                 chunk: bool = False,
                 plot_dir: pathlib.Path = pathlib.Path('/tmp')):
    """
    Uploads a loss graph, accuacy, and confusion matrix plots in addition
    to useful data about the model to Slack.

    :param x_train: the training data
    :param x_valid: the validation array
    :param y_valid: the validation labels, in the same order as x_valid
    :param model: the trained model
    :param history: the history object returned by training the model
    :param name: the name you want to give the model
    :param params: the parameters of the model to attach to the report
    :param token: your slack API token
    :param id_valid: the ids ordered to correspond with y_valid
    :param chunk: whether or not we're analyzing 3D data
    :param plot_dir: the directory to save the plots in
    :return:
    """
    os.makedirs(str(plot_dir), exist_ok=True)
    loss_path = pathlib.Path(plot_dir) / 'loss.png'
    acc_path = pathlib.Path(plot_dir) / 'acc.png'
    cm_path = pathlib.Path(plot_dir) / 'cm.png'
    tp_path = pathlib.Path(plot_dir) / 'true_positives.png'
    fp_path = pathlib.Path(plot_dir) / 'false_positives.png'
    tn_path = pathlib.Path(plot_dir) / 'true_negatives.png'
    fn_path = pathlib.Path(plot_dir) / 'false_negatives.png'

    report = _create_all_plots(x_train, x_valid, y_valid, model, history,
                               loss_path, acc_path, cm_path, tn_path, tp_path,
                               fn_path, fp_path, chunk, id_valid)

    upload_to_slack(loss_path, f'{name}\n\nparams:\n{str(params)}', token)
    upload_to_slack(acc_path, f'{name}\n\nparams:\n{str(params)}', token)
    upload_to_slack(cm_path, report, token)
    upload_to_slack(fp_path, f'{name}\n\nfalse positives', token)
    upload_to_slack(fn_path, f'{name}\n\nfalse negatives', token)
    upload_to_slack(tp_path, f'{name}\n\ntrue positives', token)
    upload_to_slack(tn_path, f'{name}\n\ntrue negatives', token)


def _create_all_plots(
        x_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        model: keras.Model,
        history: keras.callbacks.History,
        loss_path: pathlib.Path,
        acc_path: pathlib.Path,
        cm_path: pathlib.Path,
        tn_path: pathlib.Path,
        tp_path: pathlib.Path,
        fn_path: pathlib.Path,
        fp_path: pathlib.Path,
        chunk: bool = False,
        id_valid: np.ndarray = None):
    """Saves all plots to the given paths.
    """
    save_history(history, loss_path, acc_path)
    # TODO: Refactor this
    if chunk:
        y_valid = np.reshape(y_valid, (len(y_valid), 1))
        report = full_multiclass_report(model,
                                        x_valid,
                                        y_valid,
                                        classes=[0, 1],
                                        cm_path=cm_path,
                                        tp_path=tp_path,
                                        fp_path=fp_path,
                                        tn_path=tn_path,
                                        fn_path=fn_path,
                                        id_valid=id_valid,
                                        chunk=chunk)
    else:
        x_mean = np.array([x_train[:, :, :, 0].mean(),
                           x_train[:, :, :, 1].mean(),
                           x_train[:, :, :, 2].mean()])
        x_std = np.array([x_train[:, :, :, 0].std(),
                          x_train[:, :, :, 1].std(),
                          x_train[:, :, :, 2].std()])
        x_valid_standardized = (x_valid - x_mean) / x_std
        report = full_multiclass_report(model,
                                        x_valid_standardized,
                                        y_valid,
                                        classes=[0, 1],
                                        cm_path=cm_path,
                                        tp_path=tp_path,
                                        fp_path=fp_path,
                                        tn_path=tn_path,
                                        fn_path=fn_path,
                                        id_valid=id_valid,
                                        chunk=chunk)
    return report


def save_history(history: keras.callbacks.History,
                 loss_path: pathlib.Path,
                 acc_path: pathlib.Path):
    """
    Saves plots of the loss/acc over epochs in the given paths.

    :param history:
    :param loss_path:
    :param acc_path:
    :return:
    """
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
    plt.savefig(loss_path)

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
    plt.savefig(acc_path)


def save_confusion_matrix(cm, classes,
                          cm_path: pathlib.Path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints, plots, and saves the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
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

    plt.savefig(cm_path)


def full_multiclass_report(model: keras.models.Model,
                           x: np.ndarray,
                           y_true: np.ndarray,
                           classes: typing.Sequence,
                           cm_path: pathlib.Path,
                           tp_path: pathlib.Path,
                           tn_path: pathlib.Path,
                           fp_path: pathlib.Path,
                           fn_path: pathlib.Path,
                           id_valid: np.ndarray = None,
                           chunk=False):
    """
    Builds a report containing the following:
        - accuracy
        - AUC
        - classification report
        - confusion matrix
        - 7/31/2018 metrics

    The output is the report as a string.

    The report also generates a confusion matrix plot and tp/fp examples.

    :param model:
    :param x:
    :param y_true:
    :param classes:
    :param id_valid
    :param chunk
    :return:
    """
    # TODO(luke): Split this into separate functions.
    y_proba = model.predict(x, batch_size=8)
    assert y_true.shape == y_proba.shape

    if y_proba.shape[-1] == 1:
        y_pred = (y_proba > 0.5).astype('int32')
    else:
        y_pred = y_proba.argmax(axis=1)
        y_true = y_true.argmax(axis=1)

    assert y_pred.shape == y_true.shape, \
        f'y_pred.shape: {y_pred.shape} must equal y_true.shape: {y_true.shape}'

    comment = "Accuracy: " + str(
        sklearn.metrics.accuracy_score(y_true, y_pred))
    comment += '\n'

    # Assuming 0 is the negative label
    y_true_binary = y_true > 0
    y_pred_binary = y_pred > 0
    score = sklearn.metrics.roc_auc_score(y_true_binary,
                                          y_pred_binary)

    # Do not change the line below, it affects reporting._extract_auc
    comment += f'AUC: {score}\n'
    comment += f'Assuming {0} is the negative label'
    comment += '\n\n'

    comment += "Classification Report\n"
    comment += sklearn.metrics.classification_report(y_true, y_pred, digits=5)

    cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    comment += '\n'
    comment += str(cnf_matrix)
    save_confusion_matrix(cnf_matrix, classes=classes, cm_path=cm_path)

    # Compute additional metrics for the 7/31 paper
    try:
        tn, fp, fn, tp = cnf_matrix.ravel()
        comment += f'\n\nAdditional statistics:\n'
        sensitivity = tp / (tp + fn)
        comment += f'Sensitivity: {sensitivity}\n'
        specificity = tn / (tn + fp)
        comment += f'Specificity: {tn / (tn + fp)}\n'
        comment += f'Precision: {tp / (tp + fp)}\n'
        total_acc = (tp + tn) / (tp + tn + fp + fn)
        random_acc = (((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp))
                      / (tp + tn + fp + fn) ** 2)
        comment += f'\n\nNamed statistics:\n'
        kappa = (total_acc - random_acc) / (1 - random_acc)
        comment += f'Cohen\'s Kappa: {kappa}\n'
        youdens = sensitivity - (1 - specificity)
        comment += f'Youden\'s index: {youdens}\n'

        comment += f'\n\nOther sklearn statistics:\n'
        log_loss = sklearn.metrics.classification.log_loss(y_true, y_pred)
        comment += f'Log loss: {log_loss}\n'
        comment += f'F-1: {sklearn.metrics.f1_score(y_true, y_pred)}\n'
    except ValueError as e:
        comment += '\nCould not add additional statistics (tp, fp, etc.)'
        comment += str(e)

    save_misclassification_plots(x,
                                 y_true_binary,
                                 y_pred_binary,
                                 id_valid=id_valid,
                                 chunk=chunk,
                                 tp_path=tp_path,
                                 fp_path=fp_path,
                                 tn_path=tn_path,
                                 fn_path=fn_path)
    return comment


def write_to_slack(comment, token):
    """
    Write results to slack.
    """
    channels = 'CBUA09G68'

    r = requests.get(
        'https://slack.com/api/chat.postMessage?' +
        'token={}&channel={}&text={}'.format(token, channels, comment))
    return r


def write_iteration_results(params, result, slack_token,
                            job_name=None, job_date=None,
                            purported_accuracy=None,
                            purported_loss=None,
                            purported_sensitivity=None,
                            final=False, i=0):
    """
    Write iteration results (during validation) to Slack.
    """
    if final:
        text = "-----Final Results-----\n"
    else:
        text = "-----Iteration {}-----\n".format(i + 1)
    text += "Seed: {}\n".format(params.seed)
    text += "Params: {}\n".format(params)
    if (job_name is not None):
        text += 'Job name: {}\n'.format(job_name)
        text += 'Job date: {}\n'.format(job_date)
        text += 'Purported accuracy: {}\n'.format(
            purported_accuracy)
        text += 'Purported loss: {}\n'.format(
            purported_loss)
        text += 'Purported sensitivity: {}\n'.format(
            purported_sensitivity)
    if final:
        text += "\n-----Average Results-----\n"
    else:
        text += "\n-----Results-----\n"
    text += 'Loss: {}\n'.format(result[0])
    text += 'Acc: {}\n'.format(result[1])
    text += 'Sensitivity: {}\n'.format(result[2])
    text += 'Specificity: {}\n'.format(result[3])
    text += 'True Positives: {}\n'.format(result[4])
    text += 'False Negatives: {}\n'.format(result[5])
    write_to_slack(text, slack_token)


def upload_to_slack(filename,
                    comment,
                    token,
                    channels='#model-results'):
    """
    Uploads the file at the given path to the channel.

    :param filename:
    :param comment:
    :param token:
    :param channels:
    :return:
    """
    my_file = {
        'file': (str(filename), open(filename, 'rb'), 'png')
    }

    payload = {
        "filename": str(filename),
        "token": token,
        'initial_comment': comment,
        "channels": channels,
    }

    r = requests.post("https://slack.com/api/files.upload",
                      params=payload,
                      files=my_file)
    return r


def save_misclassification_plots(x_valid,
                                 y_true,
                                 y_pred,
                                 tp_path: pathlib.Path,
                                 tn_path: pathlib.Path,
                                 fp_path: pathlib.Path,
                                 fn_path: pathlib.Path,
                                 id_valid: np.ndarray = None,
                                 chunk=False):
    """Saves the 4 true/false positive/negative plots.

    The y inputs must be binary and 1 dimensional.
    """
    assert len(x_valid) == len(y_true)
    if y_true.max() > 1 or y_pred.max() > 1:
        raise ValueError('y_true/y_pred should be binary 0/1')

    plot_name_dict = {
        (0, 0): tn_path,
        (1, 1): tp_path,
        (0, 1): fp_path,
        (1, 0): fn_path,
    }

    for i in (0, 1):
        for j in (0, 1):
            mask = np.logical_and(y_true == i, y_pred == j)
            x_filtered = np.array([x_valid[i] for i, truth in enumerate(mask)
                                   if truth])

            if id_valid is None:
                ids_filtered = None
            else:
                ids_filtered = id_valid[mask]

            plot_misclassification(x_filtered,
                                   y_true[mask],
                                   y_pred[mask],
                                   ids=ids_filtered,
                                   chunk=chunk)
            plt.savefig(plot_name_dict[(i, j)])


def plot_misclassification(x,
                           y_true,
                           y_pred,
                           num_cols=5,
                           limit=20,
                           offset=0,
                           ids: np.ndarray = None,
                           chunk=False):
    """
    Plots the figures with labels and predictions.

    :param x:
    :param y_true:
    :param y_pred:
    :param num_cols:
    :param limit:
    :param offset:
    :param ids:
    :param chunk:
    :return:
    """
    num_rows = (min(len(x), limit) + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(10, 10))
    for i, arr in enumerate(x):
        if i < offset:
            continue
        if i >= offset + limit:
            break
        plot_num = i - offset + 1
        ax = fig.add_subplot(num_rows, num_cols, plot_num)
        if ids is not None:
            ax.set_title(f'patient: {ids[i][:4]}...')
        ax.set_xlabel(f'y_true: {y_true[i]} y_pred: {y_pred[i]}')
        if chunk:
            mip = np.max(arr, axis=0)
            mip = np.reshape(mip, (32, 32))
            plt.imshow(mip)
        else:
            plt.imshow(arr)  # Multiply by 255 here for
    fig.tight_layout()
    plt.plot()


def plot_images(data: typing.Dict[str, np.ndarray],
                labels: pd.DataFrame,
                num_cols=5,
                limit=20,
                offset=0):
    """
    Plots limit images in a single plot.

    :param data:
    :param labels:
    :param num_cols:
    :param limit: the number of images to plot
    :param offset:
    :return:
    """
    # Ceiling function of len(data) / num_cols
    num_rows = (min(len(data), limit) + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(10, 10))
    for i, patient_id in enumerate(data):
        if i < offset:
            continue
        if i >= offset + limit:
            break
        plot_num = i - offset + 1
        ax = fig.add_subplot(num_rows, num_cols, plot_num)
        ax.set_title(f'patient: {patient_id[:4]}...')
        label = ('positive' if labels.loc[patient_id]['occlusion_exists']
                 else 'negative')
        ax.set_xlabel(f'label: {label}')
        plt.imshow(data[patient_id])
    fig.tight_layout()
    plt.plot()
