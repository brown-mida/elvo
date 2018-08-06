"""
One-time use script on instance-1 VM to just get smaller test datasets to do
stats on for 3D model.
"""

import pickle
import numpy as np

# DATA WITH HARD NEGATIVES -- filtered
# load and unpack data separated by IDs (from etl/roi_train_preprocess.py)
with open('chunk_data_separated_ids_hard.pkl', 'rb') as infile:
    full_data = pickle.load(infile)

x_test = np.asarray(full_data[4])
y_test = np.asarray(full_data[5])

with open('test_train_hard.pkl', 'wb') as outfile:
    pickle.dump(x_test, outfile, pickle.HIGHEST_PROTOCOL)

with open('test_val_hard.pkl', 'wb') as outfile:
    pickle.dump(y_test, outfile, pickle.HIGHEST_PROTOCOL)

# DATA WITH NAIVELY PICKED NEGATIVED -- unfiltered
# load and unpack data separated by IDs (from etl/roi_train_preprocess.py)
with open('chunk_data_separated_ids.pkl', 'rb') as infile:
    full_data = pickle.load(infile)

x_test = np.asarray(full_data[4])
y_test = np.asarray(full_data[5])

with open('test_train.pkl', 'wb') as outfile:
    pickle.dump(x_test, outfile, pickle.HIGHEST_PROTOCOL)

with open('test_val.pkl', 'wb') as outfile:
    pickle.dump(y_test, outfile, pickle.HIGHEST_PROTOCOL)
