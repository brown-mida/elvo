"""
This script serves serves as a way to train the C3D model on subsets of the
training data of varying sizes, reporting the results of that training to
the model_results channel on Slack.
"""
from blueno.slack import slack_report
from blueno import utils
import tensorflow as tf
from ml.models.three_d import c3d
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import pickle
import os

BLACKLIST = []
LEARN_RATE = 1e-5

# Make sure that the GPU is being used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# load and unpack data separated by IDs (from etl/roi_train_preprocess.py)
with open('chunk_data_separated_ids.pkl', 'rb') as infile:
    full_data = pickle.load(infile)
full_x_train = full_data[0]
full_y_train = full_data[1]
x_val = full_data[2]
y_val = full_data[3]
x_test = full_data[4]
y_test = full_data[5]

assert len(full_x_train) == len(full_y_train)
assert len(x_val) == len(y_val)
assert len(x_test) == len(y_test)

metrics = ['acc',
           utils.true_positives,
           utils.false_negatives,
           utils.sensitivity,
           utils.specificity]

# for each possible fraction of the data
# for i in range(1, 11):
for i in range(4):
    for j in range(10, 11):

        # build a model
        model = c3d.C3DBuilder.build()
        opt = SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt,
                      loss={"out_class": "binary_crossentropy"},
                      metrics=metrics)

        # Downsample the training data
        frac = j / 10
        x_train = full_x_train[:int(len(full_x_train) * frac)]
        y_train = full_y_train[:int(len(full_y_train) * frac)]

        # make callbacks — AUC, sensitivity, specificity, accuracy, ModelCheckpoint
        callbacks = utils.create_callbacks(x_train=x_train,
                                           y_train=y_train,
                                           x_valid=x_val,
                                           y_valid=y_val,
                                           normalize=False)
        checkpoint = ModelCheckpoint(f'tmp/FINAL_RUN_{i}.hdf5',
                                     monitor='val_acc',
                                     verbose=1, save_best_only=True,
                                     mode='auto')
        callbacks.append(checkpoint)

        # train the model
        history = model.fit(x=x_train,
                            y=y_train,
                            epochs=100,
                            batch_size=16,
                            callbacks=callbacks,
                            validation_data=(x_val, y_val),
                            verbose=1)

        # output a slack report about how well the model trained
        slack_report(x_train=x_train,
                     x_valid=x_test,
                     y_valid=y_test,
                     model=model,
                     history=history,
                     name=f'Basic C3D (training on {frac * 100}% of data)',
                     params=f'The most basic, non-optimized version of C3D, '
                            f'training on {frac * 100}% of data',
                     token='xoxp-314216549302'
                           '-332571517623'
                           '-402064251175'
                           '-cde67240d96f69a3534e5c919ff097e7',
                     chunk=True)
