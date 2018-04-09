import numpy as np
import os
from generators.augmented_generator import AugmentedGenerator

dirname = '/home/shared/data/data-20180405'
labels = '/home/shared/data/elvos_meta_drop1.xls'

gen = AugmentedGenerator(dirname, labels, dims=(200, 200, 24), batch_size=1,
                         extend_dims=False)
while True:
    a = gen.generate()
    b, c = next(a)
    print(np.shape(b))
