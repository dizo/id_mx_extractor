# Trained model weights (not included)

Place the two files produced by `notebooks/train_ine_model.ipynb` here:

- `unet_ine.h5` — card segmentation/localization model.
- `field_detector.pth` — per-field bounding box detector.

`webapp/ine_extractor.py` checks for these files automatically at request
time: if both, either, or neither are present it uses them (falling back to
the OpenCV heuristic and/or whole-card OCR + regex for whichever is
missing). No code changes or restarts beyond dropping the files here are
needed.

These files are intentionally **not** committed to git (see `.gitignore`) —
they can be large binaries and are meant to be regenerated per-user by
running the training notebook.
