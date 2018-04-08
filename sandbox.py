import numpy as np
import os
from generators.generator import Generator

dirname = '/home/shared/data/data-20180407'
labels = '/home/shared/data/elvos_meta_drop1.xls'

gen = Generator(dirname, labels, dim_length=200, batch_size=16)
while True:
    a = gen.generate()
    b, c = next(a)
    print(np.shape(b))
