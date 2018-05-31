# Links with models / preprocessing details

[Alzheimer's Disease Diagnostics By A Deeply Supervised Adaptable 3D Convolutional Network](https://arxiv.org/pdf/1607.00556.pdf)

- Predicts Alzheimer's Disease based on 3D scan of the brain (317x215x254), similar to our inputs
- Pretrains the model using a Convolutional Autoencoder, then fine-tunes it to domain-specific (AD detection)

[V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/pdf/1606.04797.pdf)

- Uses Vnet

[Advanced machine learning in action: identification of intracranial hemorrhage on computed tomography scans of the head with clinical workflow integration](https://www.nature.com/articles/s41746-017-0015-z.pdf)

- Uses a 3D version of alexnet

[Deep Convolutional Neural Networks for Lung Cancer Detection](http://cs231n.stanford.edu/reports/2017/pdfs/518.pdf)

- Uses Unet
- Also provides detailed preprocessing details

[Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?](https://arxiv.org/pdf/1706.00712.pdf)

- Finetunes after training on Imagenet using Alexnet
- Evidence of effectiveness of transfer learning even on unrelated images

[Recurrent Convolutional Networks for Pulmonary Nodule Detection in CT Imaging](https://arxiv.org/pdf/1609.09143.pdf)

- A recurrent CNN architecture, haven't read the paper in detail yet

# Relevant links

[Deep convolutional neural networks for brain image analysis on magnetic resonance imaging: a review](https://arxiv.org/pdf/1712.03747.pdf)

- Provides a comprehensive overview of CNNs in the medical field, including practices in pre-processing.

[Detecting Cancer Metastases on Gigapixel Pathology Images](https://drive.google.com/file/d/0B1T58bZ5vYa-QlR0QlJTa2dPWVk/view)

- Detects cancer metastasis from a 2D 100,000 x 100,000 image
- Separates each image into 299x299 image "patches" to train. This requires annotating the original data on where on the original image the tumors exist.
- Uses a variation of the Inception V3 architecture (from Google)
- Although we cannot use patches, the V3 architecture could be used

[Estimating CT Image from MRI Data Using 3D Fully Convolutional Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5654583/pdf/nihms875224.pdf)

- Predicts CT images from MRI images, as the name implies
- Uses a Fully Convolutional Network (FCN, not to be confused with Fully Connected Networks)
- Extract a large number of smaller "patches" from the MRI image (instead of the entire image itself) to use as training data. This leads to a smaller (32x32x16) input and (24x24x12) output. 
- Although we cannot use patches, FCNs could be used

[Cancer Image Archive](http://www.cancerimagingarchive.net/)

Pretty much the only CT data for pretraining

[OpenfMRI](https://openfmri.org/)

Dataset of MRI data for pretraining

# Not very relevant work, but keeping for reference

[Dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/articles/nature21056.epdf?referrer_access_token=RcIKxkNHJfvxwTkuTW4lPNRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuPxB4B9GCLPvFTfGPu3BrO0euO-oKmEt01tc_3Bad0Edx-p21z_GXJAeVxTVS6U_o7mmt1TxNO3XcR6ZK9ofxEEeFaRl3oGpNMQBIFWEz9lVYs1gWvSGHUzq1_WTPh2nfp3Rx8zKAPHxpbvedE5qWe3w7F8nyvpmzbjlR_EbOOvaY%3D&tracking_referrer=www.wired.co.uk)

- Classifies the skin cancer based on the image
- Assumes skin cancer exists in the first place

# Convolutional Recursive Deep Learning

[Convolutional-Recursive Deep Learning for 3D Object Classification](https://papers.nips.cc/paper/4773-convolutional-recursive-deep-learning-for-3d-object-classification.pdf)

[Combining Fully Convolutional and Recurrent Neural Networks for 3D Biomedical Image Segmentation](https://arxiv.org/pdf/1609.01006.pdf)

[A fast and robust convolutional neural network-based defect detection model in product quality control](https://link.springer.com/content/pdf/10.1007%2Fs00170-017-0882-0.pdf)

[Convolutional Recurrent Neural Networks for Electrocardiogram Classification](https://arxiv.org/pdf/1710.06122.pdf)

[Recurrent Convolutional Neural Network for Object Recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298958&tag=1)

# Unfiled

[Improving Deep Pancreas Segmentation in CT and MRI Images via Recurrent Neural Contextual Learning and Direct Loss Function](https://arxiv.org/pdf/1707.04912.pdf)

[Holistic Interstitial Lung Disease Detection using Deep Convolutional Neural Networks: Multi-label Learning and Unordered Pooling](file:///home/rladbsgh/Downloads/Holistic_Interstitial_Lung_Disease_Detection_using.pdf)
