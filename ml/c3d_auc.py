from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import pickle
from models.three_d import c3d
from matplotlib import pyplot as plt

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with open('chunk_data_separated_ids.pkl', 'rb') as infile:
    full_data = pickle.load(infile)

x_val = full_data[2]
y_val = full_data[3]

model = c3d.C3DBuilder.build()
model.load_weights('tmp/c3d_separated_ids.hdf5')

y_pred_keras = model.predict(x_val).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
