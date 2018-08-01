"""
This script serves as a test of training a C3D model on completely unaugmented
data and seeing how it performs — preliminary results show that it hits an
accuracy of ~85%, depending on which negatives are randomly chosen. This script
is meant to be used in conjunction with etl/roi_train_preprocess_no_aug.py,
which collates the chunks to train on.
"""

import tensorflow as tf
from ml.models.three_d import c3d
from blueno.slack import slack_report
from blueno import utils
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import pickle
import os

# make sure GPU is being used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

BLACKLIST = []
LEARN_RATE = 1e-5

# load data
with open('chunk_data_no_aug.pkl', 'rb') as infile:
    full_data = pickle.load(infile)
x_train = full_data[0]
y_train = full_data[1]
x_val = full_data[2]
y_val = full_data[3]

metrics = ['acc',
           utils.true_positives,
           utils.false_negatives,
           utils.sensitivity,
           utils.specificity]

# train 10 models on the data
for i in range(10):
    model = c3d.C3DBuilder.build()
    opt = SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt,
                  loss={"out_class": "binary_crossentropy"},
                  metrics=metrics)
    callbacks = utils.create_callbacks(x_train=x_train,
                                       y_train=y_train,
                                       x_valid=x_val,
                                       y_valid=y_val,
                                       normalize=False)

    checkpoint = ModelCheckpoint(f'tmp/c3d_no_aug_{i}.hdf5',
                                 monitor='val_acc',
                                 verbose=1, save_best_only=True,
                                 mode='auto')
    callbacks.append(checkpoint)
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=100,
                        batch_size=16,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val),
                        verbose=1)
    slack_report(x_train=x_train,
                 x_valid=x_val,
                 y_valid=y_val,
                 model=model,
                 history=history,
                 name=f'Basic C3D (training on unaugmented data)',
                 params=f'The most basic, non-optimized version of C3D, '
                        f'training on unaugmented data',
                 token='xoxp-314216549302'
                       '-332571517623'
                       '-402064251175'
                       '-cde67240d96f69a3534e5c919ff097e7',
                 chunk=True)
