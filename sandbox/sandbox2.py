import numpy as np
import matplotlib.pyplot as plt
from generators.mnist_generator import MnistGenerator


gen = MnistGenerator(dims=(200, 200, 24),
                     batch_size=4).generate()
for data, label in gen:
    plt.imshow(np.reshape(data[0, :, :, 0], (200, 200)))
    plt.show()
