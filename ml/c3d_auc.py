"""
A script to plot the AUC curve for successful C3D models
"""

from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import pickle
from ml.models.three_d import c3d
from matplotlib import pyplot as plt

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# load the test data
with open('chunk_data_separated_ids.pkl', 'rb') as infile:
    full_data = pickle.load(infile)
x_test = full_data[2]
y_test = full_data[3]

# load the model
model = c3d.C3DBuilder.build()
model.load_weights('tmp/c3d_separated_ids.hdf5')

# make predictions on test data
y_pred_keras = model.predict(x_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)

# plot the ROC
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
