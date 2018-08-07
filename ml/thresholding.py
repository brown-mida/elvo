"""Script for computing a matrix of statistics at each
threshold.
"""

import keras
import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification, roc_auc_score

import blueno
from blueno.preprocessing import prepare_data
from generators.luke import standard_generators
from models.luke import resnet


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
    for i in range(0, 11):
        y_pred = model.predict(x)

        if y_pred.shape[-1] == 1:
            threshold = i / 10

            y_pred_classes = (y_pred > threshold).astype('int32')
            y_true_classes = (y_true > threshold).astype('int32')
            print('report for threshold of {}'.format(threshold))
        elif y_pred.shape[-1] == 2:
            offset = i / 10 - 0.5
            y_pred[:, 1] += offset
            y_pred_classes = y_pred.argmax(axis=1)
            y_true_classes = y_true.argmax(axis=1)
            print('\n\nreport for offset of {}'.format(offset))
            print(
                "(where a prediction y=[0.15, 0.25] is argmax{0.15, 0.25 + "
                "offset})")
        else:
            raise ValueError('Only binary models are supported')

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
    # TODO: Put the right model here
    model_path = '/home/lzhu7/elvo-analysis/models/' \
                 'processed-lower_2-classes-2018-08-02T15:31:14.880339.hdf5'
    model = blueno.io.load_model(model_path, compile=False)

    # TODO: Make sure to specify the correct data
    params = blueno.ParamConfig(
        data=blueno.DataConfig(
            data_dir='/home/lzhu7/elvo-analysis/data/processed-lower/arrays',
            labels_path='/home/lzhu7/elvo-analysis/data/processed-lower/labels'
                        '.csv',
            index_col='Anon ID',
            label_col='occlusion_exists',
            gcs_url='gs://elvos/processed/processed-lower'),
        generator=blueno.GeneratorConfig(
            generator_callable=standard_generators,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False),
        model=blueno.ModelConfig(model_callable=resnet,
                                 optimizer=Adam(lr=1e-5),
                                 loss=categorical_crossentropy,
                                 dropout_rate1=0.7,
                                 dropout_rate2=0.7, freeze=False),
        batch_size=5, seed=42, val_split=0.1, max_epochs=100,
        early_stopping=True, reduce_lr=False,
    )

    arrays = prepare_data(params, train_test_val=False)
    x_train, x_valid, y_train, y_valid, id_train, id_valid = arrays
    valid_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True)
    valid_datagen.fit(x_train)
    x_valid = valid_datagen.standardize(x_valid.astype(np.float32))
    compute_thresholds(model, x_valid, y_valid)
