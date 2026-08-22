"""UNet segmentation model that locates/crops the INE card in a photo.

Architecture mirrors the UNet already used in this repo's ``id_card.ipynb``
notebook (double conv blocks, 4 down/up stages), adapted to output a single
binary mask and to read training data lazily via ``tf.data`` instead of
loading every image into memory up front.
"""
from __future__ import annotations

import json
import os
import random
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

DEFAULT_INPUT_SIZE = 256


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def _downsample_block(x, n_filters):
    f = _double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def _upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = _double_conv_block(x, n_filters)
    return x


def build_unet_model(input_size: int = DEFAULT_INPUT_SIZE) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_size, input_size, 3))

    f1, p1 = _downsample_block(inputs, 64)
    f2, p2 = _downsample_block(p1, 128)
    f3, p3 = _downsample_block(p2, 256)
    f4, p4 = _downsample_block(p3, 512)

    bottleneck = _double_conv_block(p4, 1024)

    u6 = _upsample_block(bottleneck, f4, 512)
    u7 = _upsample_block(u6, f3, 256)
    u8 = _upsample_block(u7, f2, 128)
    u9 = _upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)

    return tf.keras.Model(inputs, outputs, name="ine_card_unet")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def _load_image_mask(image_path, mask_path, input_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (input_size, input_size))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (input_size, input_size), method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask


def build_segmentation_datasets(
    dataset_dir: str,
    input_size: int = DEFAULT_INPUT_SIZE,
    batch_size: int = 16,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Reads ``dataset_dir/annotations.json`` (as produced by
    ``synthetic_data.generate_dataset``) and returns
    (train_ds, val_ds, n_train, n_val) tf.data pipelines.
    """
    ann_path = os.path.join(dataset_dir, "annotations.json")
    with open(ann_path, encoding="utf-8") as fh:
        annotations = json.load(fh)

    image_paths = [os.path.join(dataset_dir, "images", a["image"]) for a in annotations]
    mask_paths = [os.path.join(dataset_dir, "masks", a["mask"]) for a in annotations]

    idx = list(range(len(image_paths)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_split))
    val_idx = set(idx[:n_val])

    def split(paths):
        train = [p for i, p in enumerate(paths) if i not in val_idx]
        val = [p for i, p in enumerate(paths) if i in val_idx]
        return train, val

    train_imgs, val_imgs = split(image_paths)
    train_masks, val_masks = split(mask_paths)

    def make_ds(imgs, masks, training):
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
        ds = ds.map(lambda i, m: _load_image_mask(i, m, input_size), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(buffer_size=min(1000, len(imgs)), seed=seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_imgs, train_masks, True)
    val_ds = make_ds(val_imgs, val_masks, False)
    return train_ds, val_ds, len(train_imgs), len(val_imgs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_segmentation_model(
    dataset_dir: str,
    epochs: int = 30,
    input_size: int = DEFAULT_INPUT_SIZE,
    batch_size: int = 16,
    checkpoint_path: str = "unet_ine.h5",
) -> tf.keras.Model:
    train_ds, val_ds, n_train, n_val = build_segmentation_datasets(
        dataset_dir, input_size=input_size, batch_size=batch_size
    )
    model = build_unet_model(input_size=input_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2, sparse_y_pred=False, sparse_y_true=False)],
    )
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpointer])
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_mask(model: tf.keras.Model, image_bgr: np.ndarray, input_size: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """Returns a uint8 0/255 mask resized back to the original image size."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size)).astype(np.float32) / 255.0
    pred = model.predict(resized[None, ...], verbose=0)[0, :, :, 0]
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    return cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_card_from_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Finds the largest contour in ``mask`` and perspective-warps that
    region out of ``image_bgr``. Falls back to the original image if no
    confident quadrilateral is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_bgr

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * mask.shape[0] * mask.shape[1]:
        return image_bgr

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype("float32")
    ordered = _order_points(box)
    (tl, tr, br, bl) = ordered

    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if max_width < 50 or max_height < 30:
        return image_bgr

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image_bgr, matrix, (max_width, max_height))
