
import logging
from matplotlib import pyplot as plt
from lib import transforms, cloud_management as cloud
import random
import numpy as np
import pandas as pd
import itertools

def transform_one(arr):
    iterlist = list(itertools.product('01', repeat = 4))
    axes = [[0, 1], [1, 2], [0, 2]]
    print(arr)
    for i in axes:
        rotated = np.rot90(arr, axes=i)
        print("axis = " + str(i))
        for j in iterlist:
            if j[0] == '1':
                print("vertical flip: ")
                flipped = np.flipud(arr)
            if j[1] == '1':
                print("horizontal flip: ")
                flipped = np.fliplr(arr)
            if j[2] == '1':
                print("back-front flip")
                flipped = arr[:, :, ::-1]
        print(rotated)

foo = np.array([[[0, 1]], [[0, 0]]])
transform_one(foo)