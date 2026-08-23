# Serves the INE data-extraction web app (webapp/). The synthetic dataset
# generator / training notebook (ine_model/, notebooks/) is meant to run on
# Colab/GPU and isn't part of this image — only the pieces ine_extractor.py
# needs at inference time (template.py, segmentation_model.py,
# detection_model.py) are copied in.
FROM python:3.11-slim

# Tesseract OCR (+ Spanish language data) for OCR, and the shared libraries
# opencv-python-headless needs to decode images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-spa \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

# Optional: bake in the heavier deps needed to run the trained segmentation
# (TensorFlow) / field-detection (PyTorch) models, e.g.
#   docker build --build-arg INSTALL_MODEL_DEPS=true .
# Leave this off (default) for a much smaller image that only uses the
# OpenCV heuristic + regex pipeline. Either way, dropping trained weights
# into webapp/model/ (see webapp/model/README.md) only takes effect if
# these deps are present.
ARG INSTALL_MODEL_DEPS=false
COPY webapp/requirements-model.txt webapp/requirements-model.txt
RUN if [ "$INSTALL_MODEL_DEPS" = "true" ]; then \
        pip install --no-cache-dir -r webapp/requirements-model.txt; \
    fi

COPY webapp/ webapp/
COPY ine_model/ ine_model/

WORKDIR /app/webapp

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
