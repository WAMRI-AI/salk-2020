# Crappification

# PSSR
We explore the use of original PSSR model as an advanced crappification function to generate LR images from the corresponding HR images. In this work we try to load the original model, generate predictions using the original LR images as inputs, and save predicted outputs as .tiff image files in pssr/ directory.

## Notes
The loaded PSSR model can be prepared for inference using the following code snippet:
```python
import fastai.vision as faiv

learn.path = model_path
learn.export()
learn = faiv.load_learner(learn.path, test=faiv.ImageList.from_folder(path/lr_dir), tfm_y=False)
```

Detailed code can be found in the (rather messy) notebook titled "pssr-crappification".

Tensors representing images in PyTorch are stored with dimensions (c, h, w), therefore if we want to use matplotlib for visualization we need to re-arrange the dimensions using the following code snippet:

```python
plt.imshow(tensor.permute(1,2,0))
```
Furthermore, the trained PSSR model generates predicted images that have 3 channels, while EM images are typically in gray-scale (i.e. they have only one channel). Therefore, we can alternatively visualize predictions using only the first channel of the predicted tensor. This appears to produce images with less blurring

```python
plt.imshow(tensor[0])
```

Similar to SVD crappification, a PSSR process function has been defined to generate pssr-crappified images of the original HR images. Due to the details of the trained PSSR model, the generated images were only 512x512 in size. This should be modified in the future to allow trained PSSR model to generate 600x900 images, or even images of arbitrary size at a consistent quality and resolution.

# Images Of Interest

Although it is important to acknowledge that the EM dataset consists of 99997 images of 600x900 size each. Thus, performance of any model on any task based on the dataset should primarily be assessed in terms of quantitative evaluation metrics, such as PSNR and SSIM. However, perhaps due to the inherent biases of humans to judge difference in performances based on only a few key samples, we keep a list of images to note differences between the original PSSR model and any newer versions. Such assessments should be particularly focused on the domain-specific features of the generated images i.e. identifying vesicles, etc.

Please find indexes of such key images (sorted from original 99997 images) below:
- 3500
