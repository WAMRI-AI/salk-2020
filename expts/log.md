# Experiment Archive

This document should contain an archive of all notable experiments that have been carried out as part of the Salk practicum. The requirement is generally relaxed, but a clear and brief description of each experiment should be given. The following details are of particular importance:

- name of experiment and notebook
- datasets used in the experiment (provide as a path in the Dropbox folder)
- tasks created using the dataset (classification, super-resolution, sequential...etc.)
- model architecture
- number of epochs, learning rates, and fine-tuning

Although not necessary, additional information about the reason behind the experiment, any knowledge gained, and future steps should also be included.

## Crap Critic Toy Model

### Brief:
This is an initial experiment as part of the Diversified Crappification study. Results showed initial proof-of-concept to enable a second experiment using the full dataset, possibly without pre-training.

- notebook: crap-critic-toy
- dataset: Original Salk EM images (hr), pssr generated images, SVD (k=30-40) generated images, original crappified (lr) images with upsampling to recover original size (subsample at 10%, see notebook for details)
- task: image classification (4 classes)
- model: resnet-34, pre-trained on imagenet
- training: freeze (2 cycles, 1e-3 learning rate), unfreeze (6 cycles, 3e-4 lr)

## Crap Critic ResNet Model

### Brief:
 This is a full-scale experiment as part of the Diversified Crappification study. Results showed extremely high accuracy achievable with little training. This indicates that the task might be too easy for the model to have learned useful features from the microscopy images. This has to be confirmed by using the trained model as a critic for fine-tuning the PSSR model using feature loss. However, feature loss has been originally implemented using VGG-16 as the critic's architecture. Therefore this experiment should be repeated with vgg-16 instead of resnet-34 to enable simple and consistent comparison studies.

- notebook: crap-critic-resnet34
- dataset: Original Salk EM images (hr), pssr generated images, SVD (k=30-40) generated images, original crappified (lr) images with upsampling to recover original size (no subsampling)
- task: image classification (4 classes)
- model: resnet-34, no pre-training (I think? see notebook for code details)
- model path: '/home/alaa/Dropbox/BPHO Staff/USF/EM/models/critics/full-stage-2a'
- training: freeze (2 cycles, 1e-4 learning rate)
- results: 7e-4 error rate, pretty freaking high accuracy


## Crap Critic VGG16-BN Model

### Brief:
 This is a full-scale experiment as part of the Diversified Crappification study.

- notebook: crap-critic-vgg16bn
- dataset: Original Salk EM images (hr), pssr generated images, SVD (k=30-40) generated images, original crappified (lr) images with upsampling to recover original size (no subsampling)
- task: image classification (4 classes)
- model: vgg16-bn, no pre-training (I think? see notebook for code details)
- model path: '/home/alaa/Dropbox/BPHO Staff/USF/EM/models/critics/crap-critic-vgg16bn'
- training:
- results:

## Feature Loss Baseline Model B

### Brief:
This is an initial experiment as part of the Feature Loss study. This involves fine-tuning the original PSSR model using feature loss instead of MSE loss. Feature loss is implemented using the original VGG16-BN, pre-trained on imagenet. The original PSSR model was loaded to enable rapid experimentation by using a model with already good performance. The performance of the original model is assessed in terms of evaluation metrics on the validation set prior to any fine-tuning. Then the model underwent fine-tuning training using feature loss. Results showed initial proof-of-concept to enable a second experiment using the full dataset to get a more complete assessment of the effect of feature loss on fine-tuning PSSR using VGG16-BN pre-trained on imagenet.

- notebook: feature-loss-baseline-b
- dataset: Original Salk EM images (hr), original crappified (lr) images (subsample at 10%, see notebook for details)
- task: image super-resolution
- PSSR model: original U-Net, pre-trained on original EM data using MSE loss
- Loss model: vgg16-bn, pre-trained on imagenet
- model path: '/home/alaa/Dropbox/BPHO Staff/USF/EM/models/baselines/feature_loss_baseline'
- training: freeze (1 cycle, 1e-3 learning rate), unfreeze (2 cycles, slice(1e-5,1e-3))
- results: PSNR val 11.81, 11.85, 11.88
