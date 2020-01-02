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
