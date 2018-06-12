""" The main run file.

Please use the runfile (./run) instead.
Usage: ./run -g GEN -m MOD -o OUTPUT
"""

import sys

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from ml.generators.generator import Generator
from ml.generators.mnist_generator import MnistGenerator
from ml.generators.single_generator import SingleGenerator
from ml.generators.mip_generator import MipGenerator

from ml.models.alexnet3d import AlexNet3DBuilder
from ml.models.alexnet2d import AlexNet2DBuilder
from ml.models.simple import SimpleNetBuilder

generator = sys.argv[1]
model = sys.argv[2]

# Parameters
dim_len = 120
top_len = 64
epochs = 10
batch_size = 4

# Generators
if generator == 'default':
    Gen = Generator
    num_classes = 2
    dims = 3
elif generator == 'mnist':
    Gen = MnistGenerator
    num_classes = 10
    dims = 3
elif generator == 'single':
    Gen = SingleGenerator
    num_classes = 2
    dims = 3
elif generator == 'mip':
    Gen = MipGenerator
    num_classes = 2
    dims = 2
else:
    raise ValueError('Invalid Generator')

# Model
if model == 'alexnet2d':
    Mod = AlexNet2DBuilder
elif model == 'alexnet3d':
    Mod = AlexNet3DBuilder
elif model == 'simple':
    Mod = SimpleNetBuilder
else:
    raise ValueError('Invalid Model')

# Creation
training_gen = Gen(
    batch_size=batch_size,
    augment_data=False,
    extend_dims=False
)
validation_gen = Gen(
    batch_size=batch_size,
    augment_data=False,
    extend_dims=False,
    validation=True
)

model = Mod.build((dim_len, dim_len, top_len),
                  num_classes=num_classes)
model.compile(optimizer=Adam(lr=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
mc_callback = ModelCheckpoint(filepath='tmp/alex_weights.hdf5', verbose=1)
print('Model has been compiled.')

# Training
model.fit_generator(
    generator=training_gen.generate(),
    steps_per_epoch=training_gen.get_steps_per_epoch(),
    validation_data=validation_gen.generate(),
    validation_steps=validation_gen.get_steps_per_epoch(),
    epochs=epochs,
    callbacks=[mc_callback],
    verbose=1,
    max_queue_size=1)
print('Model has been fit.')
