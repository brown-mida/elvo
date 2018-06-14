""" The main run file.

Please use the runfile (./run) instead.
Usage: ./run -g GEN -m MOD -o OUTPUT
"""

import os
import sys

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from ml.generators.generator import Generator
from ml.generators.mnist_generator import MnistGenerator
from ml.generators.single_generator import SingleGenerator
from ml.generators.multichannel_mip_generator import MipGenerator

from ml.models.alexnet3d import AlexNet3DBuilder
from ml.models.alexnet2d import AlexNet2DBuilder
from ml.models.simple import SimpleNetBuilder
from ml.models.all_conv_model import AllConvModelBuilder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parsing parameters
generator = sys.argv[1]
augment_data = sys.argv[2].upper() == 'TRUE'
extend_dims = sys.argv[3].upper() == 'TRUE'
model = sys.argv[4]
dims = [int(i) for i in sys.argv[5].split(',')]
if extend_dims:
    dims.append(1)
epochs = int(sys.argv[6])
batch_size = int(sys.argv[7])

# Generators
if generator == 'default':
    Gen = Generator
    num_classes = 2
elif generator == 'mnist':
    Gen = MnistGenerator
    num_classes = 10
elif generator == 'single':
    Gen = SingleGenerator
    num_classes = 2
elif generator == 'mip':
    Gen = MipGenerator
    num_classes = 2
else:
    raise ValueError('Invalid Generator')

# Model
if model == 'alexnet2d':
    Mod = AlexNet2DBuilder
elif model == 'alexnet3d':
    Mod = AlexNet3DBuilder
elif model == 'simple':
    Mod = SimpleNetBuilder
elif model == 'all_conv':
    Mod = AllConvModelBuilder
else:
    raise ValueError('Invalid Model')

# Creation
training_gen = Gen(
    dims=dims,
    batch_size=batch_size,
    augment_data=augment_data,
    extend_dims=extend_dims
)
validation_gen = Gen(
    dims=dims,
    batch_size=batch_size,
    augment_data=augment_data,
    extend_dims=extend_dims,
    validation=True
)

model = Mod.build(dims, num_classes=num_classes)
model.compile(optimizer=Adam(lr=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
mc_callback = ModelCheckpoint(filepath=f'tmp/{model}_weights.hdf5', verbose=1)
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
