import os
import numpy as np

loc = '/home/shared/data/data-20180405'

for filename in os.listdir(loc):
    if 'npy' in filename:
        img = np.load(loc + '/' + filename)
        if min(np.shape(img)) < 200:
            print(filename)
            print(np.shape(img))
