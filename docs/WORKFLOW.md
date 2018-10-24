# Intro to our Workflow

_Note: This is likely outdated and describes our 2018 summer workflow_.

This document assumes you have a general understanding of what machine
learning is. The [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/)
and [Machine Learning for Humans](https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12)
are good.

Our workflow consists of:
- ingesting the data from Dropbox into Google Cloud Storage
- processing the dicom files into numpy arrays ('.npy' files)
- loading the data onto a gpu machine
- training the model with Keras
- uploading reports to our team's Slack and Elasticsearch

<img src="https://i.pinimg.com/originals/0f/23/30/0f233028020bec42b6dba590a5570f45.png" width="600" />


## Model Training

Our training process involves using `ml/bluenot.py` for the following
tasks:

- hyperparameter tuning
- reporting model results

Specififying a python configuration file in the ML directory
 and then running

    ```python3 ml/bluenot.py --config=<YOUR CONFIG FILE NAME>```

will start model training on the input parameters.

Outside of `bluenot.py`, we also try out new model architectures
like VGG16 and C3D (in `ml/models`), experiment with different preprocessing
steps (slice/MIP thickness) (see `ml/bluenop.py`)

## Integrating Radiologists

Building good deep learning models often requires a good understanding
of the data and the hospital's processes.

- applying preprocessing transformations (mip, colorizing, rotation) on the data
    and viewing the result
- training "standard" models and viewing the results (seeing which
    images are misclassified)

We're planning on building a web application which replicates the
functionality of `bluenot.py` and `bluenop.py`, allowing for
radiologists