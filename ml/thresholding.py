"""One-off script for computing a matrix of "threshold" statistics.

If you wish to use this script, alter the model path and training
parameters at the bottom of the script.

Author: luke
"""
import os

import keras
import numpy as np
import pandas as pd
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification, roc_auc_score

import blueno
from blueno.preprocessing import prepare_data
from generators.luke import standard_generators

# So keras doesn't attempt to use the gpus
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_training_xy(data_root):
    # Minimal params to generate the correct training-validation data
    # split
    params = blueno.ParamConfig(
        data=blueno.DataConfig(
            data_dir='{}/arrays'.format(data_root),
            labels_path='{}/labels.csv'.format(data_root),
            index_col='Anon ID',
            label_col='occlusion_exists',
            gcs_url='gs://elvos/processed/processed-new-training-2'),
        generator=blueno.GeneratorConfig(
            generator_callable=standard_generators),
        model=blueno.ModelConfig(model_callable=None,
                                 optimizer=None,
                                 loss=categorical_crossentropy,
                                 dropout_rate1=None,
                                 dropout_rate2=None),
        batch_size=None,
        seed=0,  # or 0
        val_split=0.1,
    )
    arrays = prepare_data(params, train_test_val=False)
    return arrays[0], arrays[2]


def load_test_xy(arrays_dirpath: str, labels_filepath: str):
    array_dict = blueno.io.load_arrays(arrays_dirpath)
    index_col = 'Anon ID'
    labels_df = pd.read_csv(labels_filepath,
                            index_col=index_col)
    label_col = 'occlusion_exists'
    label_series = labels_df[label_col]
    x, y, _ = blueno.preprocessing.to_arrays(array_dict, label_series,
                                             sort=True)
    return x, y


def print_metrics(y_true, y_pred):
    print('auc:', roc_auc_score(y_true, y_pred))
    print('accuracy:', classification.accuracy_score(y_true, y_pred))

    confusion_matrix = classification.confusion_matrix(y_true, y_pred)
    # print('confusion matrix:')
    # print('report:', classification.classification_report(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix.ravel()
    sensitivity = tp / (tp + fn)
    print('sensitivity: {}'.format(sensitivity))
    specificity = tn / (tn + fp)
    print('specificity: {}'.format(specificity))
    print('precision: {}'.format(tp / (tp + fp)))
    total_acc = (tp + tn) / (tp + tn + fp + fn)
    random_acc = (((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp))
                  / (tp + tn + fp + fn) ** 2)
    kappa = (total_acc - random_acc) / (1 - random_acc)
    print('Cohen\'s kappa: {}'.format(kappa))
    youdens = sensitivity - (1 - specificity)
    print('Youden\'s index: {}'.format(youdens))
    print('log loss:', classification.log_loss(y_true, y_pred))


def compute_thresholds(model: keras.Model,
                       x: np.ndarray,
                       y_true: np.ndarray):
    y_pred = model.predict(x)

    for i in range(1, 10):
        threshold = i / 10

        if y_pred.shape[-1] == 1:
            y_pred_classes = (y_pred > threshold).astype('int32')
            y_true_classes = (y_true > threshold).astype('int32')
        elif y_pred.shape[-1] == 2:
            y_pred_classes = (y_pred[:, 1] > threshold).astype('int32')
            y_true_classes = (y_true[:, 1] > threshold).astype('int32')
        else:
            raise ValueError('Only binary models are supported')

        print('\n\nreport for threshold of {}'.format(threshold))
        y_true_binary = y_true_classes > 0
        y_pred_binary = y_pred_classes > 0
        print_metrics(y_true_binary, y_pred_binary)


def test_compute_thresholds():
    model = keras.Sequential(layers=[
        keras.layers.Flatten(input_shape=(200, 200, 3)),
        keras.layers.Dense(2, activation='softmax')
    ])
    x = np.random.rand(10, 200, 200, 3)
    y = np.random.randint(low=0, high=2, size=10)

    y_one_hot = np.array([[0, 1] if y[i] == 0 else [1, 0] for i in range(10)])
    compute_thresholds(model, x, y_one_hot)


if __name__ == '__main__':
    # Put the model you want to load here

    # gs://elvos/models/
    # processed-lower_2-classes-2018-07-13T09:59:22.804773.hdf5 .
    model_dir = '/home/lzhu7/elvo-analysis/models'
    model_name = 'processed-lower_2-classes-2018-07-13T09:59:22.804773.hdf5'
    model_path = '{}/{}'.format(model_dir, model_name)
    model = blueno.io.load_model(model_path, compile=False)

    data_root = '/home/lzhu7/elvo-analysis/data/processed-lower'
    x_train, y_train = load_training_xy(data_root)
    test_data_dir = '/home/lzhu7/elvo-analysis/data/' \
                    'processed-lower-test-155-samples'
    x_test, y_test = load_test_xy(test_data_dir + '/arrays',
                                  test_data_dir + '/labels.csv')

    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True)
    datagen.fit(x_train)
    x_train = datagen.standardize(x_train.astype(np.float32))
    x_test = datagen.standardize(x_test.astype(np.float32))
    y_test = keras.utils.to_categorical(y_test)

    print(y_train)

    compute_thresholds(model, x_train, y_train)
    compute_thresholds(model, x_test, y_test)
