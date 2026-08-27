"""Tools to generate a synthetic Mexican INE dataset and train models on it.

Nothing in this package uses or requires real INE images. It draws a
schematic (non-official-looking) card template, fills it with algorithmically
generated fake data, and composites it onto procedurally generated
backgrounds with random perspective/lighting to build a labeled dataset for:

- A segmentation model that locates/crops the card in a photo
  (``segmentation_model.py``, mirrors the UNet already used in this repo for
  the Albanian/Spanish MIDV2020 cards).
- A detection model that locates each individual field's bounding box
  (``detection_model.py``, a torchvision Faster R-CNN).

See ``notebooks/train_ine_model.ipynb`` for the end-to-end training pipeline,
meant to be run on Colab/GPU.
"""
