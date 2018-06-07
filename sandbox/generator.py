import numpy as np
from generators.generator import Generator

dirname = '/home/shared/data/data-20180405'
labels = '/home/shared/data/elvos_meta_drop1.xls'

gen = Generator(dirname, labels, dims=(200, 200, 200), batch_size=4).generate()

for each in gen:
    continue
