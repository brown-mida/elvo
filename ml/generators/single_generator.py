import os
import numpy as np

from ml.generators.generator import Generator

BLACKLIST = ['preprocess_luke/validation/LAUIHISOEZIM5ILF.npy']


class SingleGenerator(Generator):

    def __data_generation(self, i):
        bsz = self.batch_size
        files = self.files[0:bsz]
        labels = self.labels[0:bsz]
        images = []

        # Download files to tmp/npy/
        for i, file in enumerate(files):
            blob = self.bucket.get_blob(file['name'])
            file_id = file['name'].split('/')[-1]
            file_id = file_id.split('.')[0]
            blob.download_to_filename(
                'tmp/npy/{}.npy'.format(file_id)
            )
            img = np.load('tmp/npy/{}.npy'.format(file_id))
            os.remove('tmp/npy/{}.npy'.format(file_id))
            images.append(img)
        images = np.array(images)
        print("Loaded entire batch.")
        print(np.shape(images))
        return images, labels
