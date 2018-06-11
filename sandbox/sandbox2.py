import numpy as np
import matplotlib.pyplot as plt
from generators.mnist_generator import MnistGenerator


gen = MnistGenerator(dims=(200, 200, 24),
                     batch_size=4).generate()
for data, label in gen:
    fig, ax = plt.subplots(2, 2, figsize=[20, 20])
    for i in range(4):
            img = np.reshape(data[i, :, :, 0], (200, 200))
            ax[int(i / 2), int(i % 2)].imshow(img, cmap='gray')
    plt.show()
    print(label)
