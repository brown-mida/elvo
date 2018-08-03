from keras.models import Model
from keras.layers import Input, Average, Maximum
from keras.optimizers import SGD
from models.three_d import c3d
from blueno import utils
import sklearn
import pickle

LEARN_RATE = 1e-5


def ensemble(models, model_input):
    outputs = [model(model_input) for model in models]
    y_avg = Average()(outputs)
    model_avg = Model(model_input, y_avg, name='avg_ensemble')
    y_max = Maximum()(outputs)
    model_max = Model(model_input, y_max, name='max_ensemble')
    return model_avg, model_max


def get_ensembles():
    model1 = c3d.C3DBuilder.build()
    model1.load_weights('tmp/hard_training_9.hdf5')  # highest sensitivity (.947)
    model2 = c3d.C3DBuilder.build()
    model2.load_weights('tmp/hard_training_2.hdf5')  # highest specificity (.969)
    models = [model1, model2]
    model_input = Input(shape=(32, 32, 32, 1))
    ensemble_avg, ensemble_max = ensemble(models, model_input)
    metrics = ['acc',
               utils.true_positives,
               utils.false_negatives,
               utils.sensitivity,
               utils.specificity]
    opt = SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True)
    ensemble_avg.compile(optimizer=opt,
                         loss={"out_class": "binary_crossentropy"},
                         metrics=metrics)
    ensemble_max.compile(optimizer=opt,
                         loss={"out_class": "binary_crossentropy"},
                         metrics=metrics)
    return [ensemble_avg, ensemble_max]


def get_test_chunks():
    with open('test_train_hard.pkl', 'rb') as infile:
        test_train_hard = pickle.load(infile)
    with open('test_val_hard.pkl', 'rb') as infile:
        test_val_hard = pickle.load(infile)
    return test_train_hard, test_val_hard


def make_preds(x_test, y_test, models):
    for i, model in enumerate(models):
        if i % 2 == 0:
            print('\n-----------\nAVERAGE\n-----------\n')
        else:
            print('\n-----------\nMAXIMUM\n-----------\n')
        y_prob = model.predict(x_test, batch_size=16)
        y_pred = (y_prob > 0.5).astype('int32')
        print("Accuracy: " + str(
            sklearn.metrics.accuracy_score(y_test, y_pred)))
        y_true_binary = y_test > 0
        y_pred_binary = y_pred > 0
        score = sklearn.metrics.roc_auc_score(y_true_binary,
                                              y_pred_binary)
        print(f"AUC: {score} (assuming 0 negative label)")
        cnf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

        print(f'Confusion matrix:\n{str(cnf_matrix)}')
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

        print(f'\n\nOther sklearn statistics:\n')
        log_loss = sklearn.metrics.classification.log_loss(y_test, y_pred)
        print(f'Log loss: {log_loss}\n')
        print(f'F-1: {sklearn.metrics.f1_score(y_true, y_pred)}\n')


def main():
    models = get_ensembles()
    x_test, y_test = get_test_chunks()
    make_preds(x_test, y_test, models)


if __name__ == '__main__':
    main()
