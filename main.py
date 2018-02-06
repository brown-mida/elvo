import os

import numpy as np
import pandas as pd
import scipy.ndimage

import transforms


def load_scan(dirpath):
    """Takes in the path of a directory containing scans and
    returns a list of dicom dataset objects. Each dicom dataset
    contains a single image slice.
    """
    slices = [transforms.dicom.read_file(dirpath + '/' + filename)
              for filename in transforms.os.listdir(dirpath)]
    return sorted(slices, key=lambda x: float(x.ImagePositionPatient[2]))


def load_scans(input_dir):
    id_pattern = transforms.re.compile(r'\d+')
    patient_ids = []
    preprocessed_scans = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if filenames and '.dcm' in filenames[0]:
            patient_id = id_pattern.findall(dirpath)[0]
            patient_ids.append(patient_id)
            preprocessed_scans.append(transforms.load_scan(dirpath))
            print('Loaded data for patient', patient_id)
    return patient_ids, preprocessed_scans


def preprocess(input_dir, output_dir):
    """Loads the data from the input directory and saves
    normalized, zero-centered, 200 x 200 x 200 3D renderings
    in output_dir.
    """
    # Input (NOTE: Keep I/O separate from transformations for clarity!)
    patient_ids, preprocessed_scans = load_scans(input_dir)

    # Transformations
    processed_scans = []
    for id_, slices in zip(patient_ids, preprocessed_scans):
        scan = transforms.get_pixels_hu(slices)
        scan = transforms.standardize_spacing(scan, slices)
        scan = transforms.crop(scan)
        # TODO: Generate an image at this point to verify the preprocessing
        print('Finished converting to HU, standardizing pixels'
              ' to 1mm, and cropping the array to 200x200x200')
        processed_scans.append(scan)

    # Consider doing this step just before training the model
    # normalized = normalize(np.stack(processed_scans))
    # Output
    # TODO: Consider moving this step into a separate function
    transforms.os.makedirs(output_dir)
    for id_, scan in zip(patient_ids, processed_scans):
        outfile = '{}/patient-{}'.format(output_dir, id_)
        np.save(outfile, scan)

    labels = pd.DataFrame(index=patient_ids)
    labels['label'] = 0
    for name in transforms.os.listdir(input_dir):
        if name.isdigit():
            labels.loc[name, 'label'] = 1
    labels.to_csv('{}/labels.csv'.format(output_dir))


def build_and_train():
    # Reading in the data
    images = []
    for filename in sorted(os.listdir('data-1517884138')):
        if 'csv' in filename:
            continue
        images.append(np.load('data-1517884138' + '/' + filename))

    labels = pd.read_csv('data-1517884138/labels.csv', index_col=0)
    labels.sort_index(inplace=True)

    resized = np.stack([scipy.ndimage.interpolation.zoom(arr, 32 / 200)
                        for arr in images])

    normalized = transforms.normalize(resized)

    from torch import nn
    class Flatten(nn.Module):
        def forward(self, x):
            x = x.view(-1)
            return x

    model = nn.Sequential(nn.Conv3d(1, 10, 5),
                          nn.ReLU(),
                          nn.Conv3d(10, 20, 5),
                          nn.ReLU(),
                          Flatten(),
                          nn.Linear(276480, 1000),
                          nn.Linear(1000, 2),
                          nn.Softmax(dim=0))
    model.train()

    from torch.optim import SGD
    from torch.nn import BCELoss
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = BCELoss()

    import torch
    from torch.autograd import Variable
    def train():
        for image3d, label in zip(normalized, labels['label'].values):
            image3d = Variable(torch.Tensor(image3d).unsqueeze(0).unsqueeze(0))
            true_label = Variable(torch.Tensor([int(label == 0), int(label == 1)]))
            optimizer.zero_grad()
            pred_label = model(image3d)
            loss = criterion(pred_label, true_label)
            loss.backward()
            optimizer.step()

        print('Current loss:', loss)

    for _ in range(10):
        train()

    for image3d in normalized:
        print(model(images))


if __name__ == '__main__':
    IN_DIR = 'RI Hospital ELVO Data'  # The relative path to the dataset
    OUT_DIR = 'data-{}'.format(int(transforms.time.time()))
    # preprocess(IN_DIR, OUT_DIR)
    build_and_train()