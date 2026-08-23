# ID-Card-information-extraction
An application for segmentation and recognition of information printed on identification cards. **[Bachelors dissertation (2021 December – Present)]**
- Using convolutional neural network architecture UNet for segmentation of identity card from the query image and OCR ing the segmented image using text recognition engine tesseract.
- Developing an environment for experiments with different solutions of selected steps in the whole end-to-end solution. 
## Dataset
- MIDV2020
- It consists of photos, scans and videos of 1000 different ID cards of 10 different countries which has variable text fields and unique artificially generated faces
- The information presented on these ID cards are not actual information of people and it is so close to real ID cards which makes them more applicable for achieving perfect verification without having to deal with original ID cards. 
- This work mainly focuses on Albanian and Spanish ID card types.

## Web interface (Mexican INE)
The `webapp/` directory contains a standalone Flask web app to extract data
from Mexican INE (voter ID) credentials via image preprocessing + OCR. See
[`webapp/README.md`](webapp/README.md) for setup and usage.

Quickest way to run it is Docker, from the repo root:
```bash
docker compose up --build
```
then open `http://localhost:5000`. See [`webapp/README.md`](webapp/README.md#docker)
for details (including how to enable the trained models below).

## Training INE-specific models
`ine_model/` generates a synthetic INE dataset (no real INE images —
Faker-generated fake data rendered onto a schematic card template, composited
onto random backgrounds with perspective/lighting augmentation, the same
overall recipe as the MIDV2020 pipeline above) and trains two models on it: a
UNet to locate/crop the card, and a Faster R-CNN to detect each field's
bounding box for per-field OCR. Run `notebooks/train_ine_model.ipynb` (Colab/
GPU recommended) to generate the dataset and train both models; the web app
in `webapp/` picks up the resulting weights automatically if you drop them in
`webapp/model/` — see [`webapp/model/README.md`](webapp/model/README.md).
