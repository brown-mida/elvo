import numpy as np
import os

dirname = '/home/shared/data/data-20180405'

for filename in os.listdir(dirname):
    tmp = np.load(dirname + '/' + filename)
    print(np.shape(tmp))
