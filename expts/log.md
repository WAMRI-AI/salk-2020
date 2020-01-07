# Experiment Archive

This document should contain an archive of all notable experiments that have been carried out as part of the Salk practicum. The requirement is generally relaxed, but a clear and brief description of each experiment should be given. The following details are of particular importance:

- name of experiment and notebook
- datasets used in the experiment (provide as a path in the Dropbox folder)
- tasks created using the dataset (classification, super-resolution, sequential...etc.)
- model architecture 
- number of epochs, learning rates, and fine-tuning

Although not necessary, additional information about the reason behind the experiment, any knowledge gained, and future steps should also be included. 

## Crap Critic Toy Model
- notebook: crap-critic-toy
- dataset: Original Salk EM images (hr), pssr generated images, SVD (k=30-40) generated images, original crappified (lr) images with upsampling to recover original size (subsample at 10%, see notebook for details)
- task: image classification (4 classes)
- model: resnet-34, pre-trained on imagenet
- training: freeze (2 cycles, 1e-3 learning rate), unfreeze (6 cycles, 3e-4 lr)

N.B: this was an initial experiment as part of the diversified crappification study. Results showed initial proof-of-concept to enable a second experiment using the full dataset, possibly without pre-training. 