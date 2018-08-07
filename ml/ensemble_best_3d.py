"""
This script ensembles together the C3D models with the highest sensitivity and
specificity and prints out their stats on the test set. Put together, they
outperform the mean model and have an accuracy and sensitivity higher than the
upper range of the 95% CI. MEANT TO BE RUN ON HAROLD_TRIEDMAN@INSTANCE-1.
"""

from keras.models import Model
from keras.layers import Input, Average, Maximum
from keras.optimizers import SGD
from models.three_d import c3d
from blueno import utils
import sklearn
import pickle

LEARN_RATE = 1e-5


def ensemble(models, model_input):
    """
    Actually creates and returns the ensemble models
    :param models: Which models to ensemble
    :param model_input: Input layer
    :return: average ensemble, max ensemble
    """
    outputs = [model(model_input) for model in models]
    # Average the outputs
    y_avg = Average()(outputs)
    model_avg = Model(model_input, y_avg, name='avg_ensemble')
    # Take max of the outputs
    y_max = Maximum()(outputs)
    model_max = Model(model_input, y_max, name='max_ensemble')
    return model_avg, model_max


def get_ensembles():
    """
    Load high performing models, ensemble them, then compile the ensembles
    :return: An array of the compiled models
    """
    # Load models
    model1 = c3d.C3DBuilder.build()
    model1.load_weights('tmp/hard_training_9.hdf5')  # sensitivity = .947
    model2 = c3d.C3DBuilder.build()
    model2.load_weights('tmp/hard_training_2.hdf5')  # specificity = .969
    models = [model1, model2]
    model_input = Input(shape=(32, 32, 32, 1))
    # Ensemble models
    ensemble_avg, ensemble_max = ensemble(models, model_input)
    metrics = ['acc',
               utils.true_positives,
               utils.false_negatives,
               utils.sensitivity,
               utils.specificity]
    opt = SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True)
    # Compile and return models
    ensemble_avg.compile(optimizer=opt,
                         loss='binary_crossentropy',
                         metrics=metrics)
    ensemble_max.compile(optimizer=opt,
                         loss='binary_crossentropy',
                         metrics=metrics)
    return [ensemble_avg, ensemble_max]


def get_test_chunks():
    """
    Loads the harder test set of chunks and labels
    :return: Test chunks and labels
    """
    with open('test_train_hard.pkl', 'rb') as infile:
        test_train_hard = pickle.load(infile)
    with open('test_val_hard.pkl', 'rb') as infile:
        test_val_hard = pickle.load(infile)
    return test_train_hard, test_val_hard


def make_preds(x_test, y_test, models):
    """
    Actually makes the predictions and prints the stats about how the models
    are performing.
    :param x_test: Test set data
    :param y_test: Test set labels
    :param models: Array of models to print statistics of
    :return:
    """
    for i, model in enumerate(models):
        if i == 1:
            print('\n-------\nAVERAGE\n-------\n')
        else:
            print('\n-------\nMAXIMUM\n-------\n')
        # Predict on the test set
        y_prob = model.predict(x_test, batch_size=16)
        # Cast probabilities to 0s and 1s
        y_pred = (y_prob > 0.5).astype('int32')
        # Get and print accuracy
        print("Accuracy: " + str(
            sklearn.metrics.accuracy_score(y_test, y_pred)))
        y_true_binary = y_test > 0
        y_pred_binary = y_pred > 0
        # Get and print AUC
        score = sklearn.metrics.roc_auc_score(y_true_binary,
                                              y_pred_binary)
        print(f"AUC: {score} (assuming 0 negative label)")
        # Get and print confusion matrix
        cnf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
        print(f'Confusion matrix:\n{str(cnf_matrix)}')
        # Get and print sensitivity, specificity,
        #   precision, kappa, and youden's
        tn, fp, fn, tp = cnf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        print(f'Sensitivity: {sensitivity}\n')
        specificity = tn / (tn + fp)
        print(f'Specificity: {tn / (tn + fp)}\n')
        print(f'Precision: {tp / (tp + fp)}\n')
        total_acc = (tp + tn) / (tp + tn + fp + fn)
        random_acc = (((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp))
                      / (tp + tn + fp + fn))
        print(f'\n\nNamed statistics:\n')
        kappa = (total_acc - random_acc) / (1 - random_acc)
        print(f'Cohen\'s Kappa: {kappa}\n')
        youdens = sensitivity - (1 - specificity)
        print(f'Youden\'s index: {youdens}\n')
        # Get and print log loss and F1 score
        print(f'\n\nOther sklearn statistics:\n')
        log_loss = sklearn.metrics.classification.log_loss(y_test, y_pred)
        print(f'Log loss: {log_loss}\n')
        print(f'F-1: {sklearn.metrics.f1_score(y_test, y_pred)}\n')


def main():
    models = get_ensembles()
    x_test, y_test = get_test_chunks()
    make_preds(x_test, y_test, models)
    models[0].save('tmp/ensembled_3d.hdf5')


if __name__ == '__main__':
    main()
