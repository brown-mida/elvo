from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import numpy as np

from keras.models import load_model

from generators.augmented_generator import AugmentedGenerator

# Parameters
dim_len = 200
top_len = 24
epochs = 10
batch_size = 32

data_loc = '/home/shared/data/data-20180405'
label_loc = '/home/shared/data/elvos_meta_drop1.xls'

# Generators
gen = AugmentedGenerator(data_loc, label_loc,
                         dims=(dim_len, dim_len, top_len),
                         batch_size=batch_size,
                         extend_dims=False,
                         validation=True,
                         split=0.1)

print("Validation = True")
# Build model
model = load_model('tmp/alex_weights.hdf5')
print("Model loaded")
print(str(gen.get_steps_per_epoch()) + " steps needed")
result = model.predict_generator(
    generator=gen.generate(),
    steps=3,
    max_queue_size=3)
result = np.argmax(result, axis=1)
np.save("results/pred.npy", result)
# result = np.load("results/pred.npy")
true_labels = gen.get_labels()
true_labels = true_labels[:len(result)]
print(result)
print(gen.get_labels())
print(len(result))
print(len(true_labels))
auc = roc_auc_score(true_labels, result)
print(auc)
target_names = ['No', 'Yes']
report = classification_report(true_labels, result, target_names=target_names)
print(report)
