import numpy as np
import os

dirname = '/home/shared/data/data-20180407'

for filename in os.listdir(dirname):
    if '.npy' in filename:
        tmp = np.load(dirname + '/' + filename)
        if (min(np.shape(tmp)) < 200):
            print(filename)
            print(np.shape(tmp))
