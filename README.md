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
