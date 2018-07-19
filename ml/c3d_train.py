import tensorflow as tf
from models.three_d import c3d
from blueno.slack import slack_report
from blueno import utils
from keras.optimizers import SGD
import pickle

BLACKLIST = []
LEARN_RATE = 1e-5

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with open('chunk_data.pkl', 'rb') as infile:
    full_data = pickle.load(infile)

full_x_train = full_data[0]
full_y_train = full_data[1]
full_x_val = full_data[2]
full_y_val = full_data[3]

metrics = ['acc',
           utils.true_positives,
           utils.false_negatives,
           utils.sensitivity,
           utils.specificity]

for i in range(1, 11):

    model = c3d.C3DBuilder.build()
    opt = SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt,
                  loss={"out_class": "binary_crossentropy"},
                  metrics=metrics)

    frac = i / 10
    X_train = full_x_train[:int(len(full_x_train) * frac)]
    y_train = full_y_train[:int(len(full_y_train) * frac)]
    X_val = full_x_val[:int(len(full_x_val) * frac)]
    y_val = full_y_val[:int(len(full_y_val) * frac)]
    callbacks = utils.create_callbacks(x_train=X_train,
                                       y_train=y_train,
                                       x_valid=X_val,
                                       y_valid=y_val,
                                       normalize=False)

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=100,
                        batch_size=16,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        verbose=1)

    slack_report(x_train=X_train,
                 x_valid=X_val,
                 y_valid=y_val,
                 model=model,
                 history=history,
                 name=f'Basic C3D (training on {frac * 100}% of data)',
                 params=f'The most basic, non-optimized version of C3D, '
                        f'training on {frac * 100}% of data',
                 token='xoxp-314216549302'
                       '-332571517623'
                       '-402064251175'
                       '-cde67240d96f69a3534e5c919ff097e7')
