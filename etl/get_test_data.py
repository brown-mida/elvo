import pickle
import numpy as np

# load and unpack data separated by IDs (from etl/roi_train_preprocess.py)
with open('chunk_data_separated_ids_hard.pkl', 'rb') as infile:
    full_data = pickle.load(infile)

x_test = full_data[4]
y_test = full_data[5]

test = np.array([x_test, y_test])

with open('test_data_hard.pkl', 'wb') as outfile:
    pickle.dump(test, outfile, pickle.HIGHEST_PROTOCOL)

# load and unpack data separated by IDs (from etl/roi_train_preprocess.py)
with open('chunk_data_separated_ids.pkl', 'rb') as infile:
    full_data = pickle.load(infile)

x_test = full_data[4]
y_test = full_data[5]

test = np.array([x_test, y_test])

with open('test_data.pkl', 'wb') as outfile:
    pickle.dump(test, outfile, pickle.HIGHEST_PROTOCOL)
