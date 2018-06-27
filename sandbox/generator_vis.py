import numpy as np
from ml.generators.mip_generator import MipGenerator

# gen = MipGenerator(
#     dims=(220, 220, 3),
#     batch_size=2,
#     augment_data=False,
#     extend_dims=False
# )

# for data, labels in gen.generate():
#     np.save('tmp/npy/test.npy', data)
#     throw

test = np.load('tmp/npy/test.npy')
print(np.shape(test))
