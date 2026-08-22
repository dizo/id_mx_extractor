"""Faster R-CNN field detector: given a (cropped) INE card image, locate the
bounding box of each individual printed field (NOMBRE, CURP, etc.) so the
web app can OCR each region separately instead of the whole card at once.

Uses a MobileNetV3 backbone (``fasterrcnn_mobilenet_v3_large_320_fpn``) for
faster training/inference on modest hardware; swap for
``fasterrcnn_resnet50_fpn`` in ``build_detection_model`` if you have a GPU
and want higher accuracy.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .template import FIELD_NAMES

# Label 0 is reserved for background (torchvision convention).
LABEL_TO_ID: Dict[str, int] = {name: i + 1 for i, name in enumerate(FIELD_NAMES)}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(FIELD_NAMES) + 1


class IneFieldDataset(Dataset):
    """Reads a dataset produced by ``synthetic_data.generate_dataset``."""

    def __init__(self, dataset_dir: str, annotations: Optional[List[dict]] = None):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        if annotations is None:
            with open(os.path.join(dataset_dir, "annotations.json"), encoding="utf-8") as fh:
                annotations = json.load(fh)
        # Faster R-CNN needs at least one box per image.
        self.annotations = [a for a in annotations if a["boxes"]]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        entry = self.annotations[idx]
        img_path = os.path.join(self.images_dir, entry["image"])
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(img_rgb).permute(2, 0, 1)

        boxes = torch.tensor([b["bbox"] for b in entry["boxes"]], dtype=torch.float32)
        labels = torch.tensor([LABEL_TO_ID[b["label"]] for b in entry["boxes"]], dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_dataloaders(
    dataset_dir: str, batch_size: int = 4, val_split: float = 0.15, seed: int = 42, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    dataset = IneFieldDataset(dataset_dir)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_detection_model(num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True):
    weights = "DEFAULT" if pretrained_backbone else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_detection_model(
    dataset_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.005,
    device: Optional[str] = None,
    checkpoint_path: str = "field_detector.pth",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(dataset_dir, batch_size=batch_size)

    model = build_detection_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        lr_scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        # torchvision's detection models only report losses in train() mode,
        # so we compute a comparable "validation loss" the same way instead
        # of switching to eval() for this step.
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss_dict.values()).item()
        val_loss /= max(1, len(val_loader))

        print(f"epoch {epoch + 1}/{epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_detection_model(checkpoint_path: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detection_model(pretrained_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_fields(
    model, image_bgr: np.ndarray, score_threshold: float = 0.5, device: Optional[str] = None
) -> Dict[str, Tuple[int, int, int, int]]:
    """Runs the detector and returns, per field label, the highest-scoring
    box above ``score_threshold`` as (xmin, ymin, xmax, ymax).
    """
    device = device or next(model.parameters()).device
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)

    model.eval()
    with torch.no_grad():
        output = model([tensor])[0]

    best_per_label: Dict[str, Tuple[float, Tuple[int, int, int, int]]] = {}
    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    for box, label_id, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        label = ID_TO_LABEL.get(int(label_id))
        if label is None:
            continue
        if label not in best_per_label or score > best_per_label[label][0]:
            x1, y1, x2, y2 = box
            best_per_label[label] = (float(score), (int(x1), int(y1), int(x2), int(y2)))

    return {label: box for label, (_, box) in best_per_label.items()}
