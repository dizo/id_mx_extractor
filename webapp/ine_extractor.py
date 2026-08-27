"""OCR-based extraction of printed fields from Mexican INE (voter ID) cards.

This module does not rely on a trained segmentation model (none is shipped
in this repository). Instead it uses classic image preprocessing (OpenCV)
to locate and clean up the card, Tesseract OCR to read the text, and
label-anchored regular expressions tuned to the layout of the modern INE
credential to pull out individual fields.

OCR on real-world photos is never perfect, so every field returned here is a
best-effort guess. The web UI renders the results in editable inputs so the
user can correct mistakes before using the data.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pytesseract

OCR_LANG = "spa"
DEFAULT_TESS_CONFIG = "--oem 3 --psm 6"
MRZ_TESS_CONFIG = (
    "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
)

# Optional trained models (ine_model/, see notebooks/train_ine_model.ipynb).
# Everything below degrades gracefully to the heuristic/regex pipeline above
# when tensorflow/torch aren't installed or the weight files don't exist —
# none of this is required to run the app.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
SEG_MODEL_PATH = os.path.join(MODEL_DIR, "unet_ine.h5")
DET_MODEL_PATH = os.path.join(MODEL_DIR, "field_detector.pth")

_seg_model = None
_seg_load_attempted = False
_det_model = None
_det_load_attempted = False


def _get_segmentation_model():
    global _seg_model, _seg_load_attempted
    if _seg_load_attempted:
        return _seg_model
    _seg_load_attempted = True
    if os.path.isfile(SEG_MODEL_PATH):
        try:
            import tensorflow as tf

            _seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)
        except Exception:
            _seg_model = None
    return _seg_model


def _get_detection_model():
    global _det_model, _det_load_attempted
    if _det_load_attempted:
        return _det_model
    _det_load_attempted = True
    if os.path.isfile(DET_MODEL_PATH):
        try:
            from ine_model.detection_model import load_detection_model

            _det_model = load_detection_model(DET_MODEL_PATH, device="cpu")
        except Exception:
            _det_model = None
    return _det_model


class ExtractionError(Exception):
    """Raised when the input image can't be read or processed."""


# ---------------------------------------------------------------------------
# Image loading & preprocessing
# ---------------------------------------------------------------------------

def _read_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ExtractionError("No se pudo leer la imagen. Verifica que el archivo no esté dañado.")
    return img


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_and_warp_card(img: np.ndarray) -> np.ndarray:
    """Best-effort detection + perspective correction of the ID card.

    Falls back to the original image whenever a confident quadrilateral
    covering a large-enough portion of the frame isn't found, so a tightly
    cropped photo of just the card still works fine.
    """
    h, w = img.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img.copy()
    if scale >= 1:
        scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    card_contour = None
    small_area = small.shape[0] * small.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * small_area:
            card_contour = approx
            break

    if card_contour is None:
        return img

    pts = card_contour.reshape(4, 2).astype("float32") / scale
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    if max_width < 200 or max_height < 120:
        return img

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (max_width, max_height))


def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) < 1600:
        factor = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh


def _ocr(img: np.ndarray, config: str = DEFAULT_TESS_CONFIG) -> str:
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG, config=config)
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionError(
            "Tesseract OCR no está instalado en el servidor. "
            "Instala el paquete 'tesseract-ocr' y el idioma 'spa'."
        ) from exc


def _strip_accents_upper(text: str) -> str:
    text = text.upper()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Front-side field parsing
# ---------------------------------------------------------------------------

_LABELS = [
    ("nombre", r"NOMBRE"),
    ("domicilio", r"DOMICILIO"),
    ("clave_elector", r"CLAVE\s*DE\s*ELECTOR"),
    ("curp", r"\bCURP\b"),
    ("fecha_nacimiento", r"FECHA\s*DE\s*NACIMIENTO"),
    ("sexo", r"\bSEXO\b"),
    ("anio_registro", r"A[NÑ]O\s*DE\s*REGISTRO"),
    ("estado", r"\bESTADO\b"),
    ("municipio", r"\bMUNICIPIO\b"),
    ("localidad", r"\bLOCALIDAD\b"),
    ("seccion", r"SECCI[OÓ]N"),
    ("emision", r"EMISI[OÓ]N"),
    ("vigencia", r"VIGENCIA"),
]

_CURP_RE = re.compile(r"[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d")
_CLAVE_ELECTOR_RE = re.compile(r"\b[A-Z0-9]{18}\b")
_DATE_RE = re.compile(r"(\d{2})\s*[\/\.\-]\s*(\d{2})\s*[\/\.\-]\s*(\d{4})")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_VIGENCIA_RE = re.compile(r"\b(20\d{2})\s*[-\/]\s*(20\d{2})\b|\b(20\d{2})\b")
_SECTION_RE = re.compile(r"\b\d{3,4}\b")


def _segment_by_labels(text_norm: str) -> dict:
    """Split normalized OCR text into per-label spans, in order of appearance."""
    hits = []
    for key, pattern in _LABELS:
        m = re.search(pattern, text_norm)
        if m:
            hits.append((m.start(), m.end(), key))
    hits.sort()

    segments = {}
    for i, (_, end, key) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text_norm)
        segments[key] = text_norm[end:next_start].strip(" :\n")
    return segments


def _clean_text(segment: str) -> str:
    segment = re.sub(r"[^A-Z0-9ÑÁÉÍÓÚ\s]", " ", segment)
    segment = re.sub(r"\s+", " ", segment).strip()
    return segment


def _first_line(segment: str, max_words: int = 6) -> Optional[str]:
    cleaned = _clean_text(segment)
    if not cleaned:
        return None
    words = cleaned.split(" ")[:max_words]
    return " ".join(words) if words else None


@dataclass
class FrontFields:
    nombre: Optional[str] = None
    domicilio: Optional[str] = None
    clave_elector: Optional[str] = None
    curp: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    sexo: Optional[str] = None
    anio_registro: Optional[str] = None
    estado: Optional[str] = None
    municipio: Optional[str] = None
    localidad: Optional[str] = None
    seccion: Optional[str] = None
    emision: Optional[str] = None
    vigencia: Optional[str] = None
    raw_text: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def parse_ine_front(raw_text: str) -> FrontFields:
    text_norm = _strip_accents_upper(raw_text)
    segments = _segment_by_labels(text_norm)
    fields = FrontFields(raw_text=raw_text)

    fields.nombre = _first_line(segments.get("nombre", ""), max_words=8)
    fields.domicilio = _first_line(segments.get("domicilio", ""), max_words=12)

    curp_match = _CURP_RE.search(re.sub(r"\s+", "", text_norm))
    fields.curp = curp_match.group(0) if curp_match else None

    clave_segment = re.sub(r"[^A-Z0-9]", "", segments.get("clave_elector", ""))
    clave_match = _CLAVE_ELECTOR_RE.search(clave_segment) or _CLAVE_ELECTOR_RE.search(
        re.sub(r"[^A-Z0-9]", "", text_norm)
    )
    fields.clave_elector = clave_match.group(0) if clave_match else None

    fecha_match = _DATE_RE.search(segments.get("fecha_nacimiento", "")) or _DATE_RE.search(text_norm)
    fields.fecha_nacimiento = "/".join(fecha_match.groups()) if fecha_match else None

    sexo_match = re.search(r"\b([HM])\b", segments.get("sexo", ""))
    fields.sexo = sexo_match.group(1) if sexo_match else None

    anio_match = _YEAR_RE.search(segments.get("anio_registro", ""))
    fields.anio_registro = anio_match.group(0) if anio_match else None

    fields.estado = _first_line(segments.get("estado", ""), max_words=3)
    fields.municipio = _first_line(segments.get("municipio", ""), max_words=3)
    fields.localidad = _first_line(segments.get("localidad", ""), max_words=4)

    seccion_match = _SECTION_RE.search(segments.get("seccion", ""))
    fields.seccion = seccion_match.group(0) if seccion_match else None

    emision_match = _YEAR_RE.search(segments.get("emision", ""))
    fields.emision = emision_match.group(0) if emision_match else None

    vigencia_match = _VIGENCIA_RE.search(segments.get("vigencia", ""))
    if vigencia_match:
        fields.vigencia = "-".join(g for g in vigencia_match.groups() if g)

    return fields


def _locate_card(img: np.ndarray) -> np.ndarray:
    """Crops the card out of the photo, preferring the trained segmentation
    model (ine_model/) over the OpenCV contour heuristic when available.
    """
    seg_model = _get_segmentation_model()
    if seg_model is not None:
        try:
            from ine_model.segmentation_model import DEFAULT_INPUT_SIZE, crop_card_from_mask, predict_mask

            mask = predict_mask(seg_model, img, input_size=DEFAULT_INPUT_SIZE)
            cropped = crop_card_from_mask(img, mask)
            if cropped is not img:
                return cropped
        except Exception:
            pass
    return _find_and_warp_card(img)


def _clean_field_value(label: str, text: str) -> Optional[str]:
    """Applies the same per-field regex/cleanup used by the label-based
    parser, but to a single OCR'd crop instead of a shared text segment.
    """
    text_norm = _strip_accents_upper(text)

    if label == "curp":
        m = _CURP_RE.search(re.sub(r"\s+", "", text_norm))
        return m.group(0) if m else _clean_text(text_norm) or None
    if label == "clave_elector":
        m = _CLAVE_ELECTOR_RE.search(re.sub(r"[^A-Z0-9]", "", text_norm))
        return m.group(0) if m else _clean_text(text_norm) or None
    if label == "fecha_nacimiento":
        m = _DATE_RE.search(text_norm)
        return "/".join(m.groups()) if m else _clean_text(text_norm) or None
    if label == "sexo":
        m = re.search(r"\b([HM])\b", text_norm)
        return m.group(1) if m else _clean_text(text_norm) or None
    if label in ("anio_registro", "emision"):
        m = _YEAR_RE.search(text_norm)
        return m.group(0) if m else _clean_text(text_norm) or None
    if label == "seccion":
        m = _SECTION_RE.search(text_norm)
        return m.group(0) if m else _clean_text(text_norm) or None
    if label == "vigencia":
        m = _VIGENCIA_RE.search(text_norm)
        return "-".join(g for g in m.groups() if g) if m else _clean_text(text_norm) or None
    if label in ("nombre", "domicilio"):
        return _first_line(text_norm, max_words=12)
    return _clean_text(text_norm) or None


def _extract_fields_with_detector(card_bgr: np.ndarray, det_model) -> dict:
    from ine_model.detection_model import predict_fields
    from ine_model.template import FIELD_NAMES

    boxes = predict_fields(det_model, card_bgr, score_threshold=0.5, device="cpu")
    h, w = card_bgr.shape[:2]
    pad = 4

    result = {name: None for name in FIELD_NAMES}
    raw_parts = []
    for label, (x1, y1, x2, y2) in boxes.items():
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = card_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        prepped_crop = _preprocess_for_ocr(crop)
        psm = "7" if label not in ("nombre", "domicilio") else "6"
        text = _ocr(prepped_crop, config=f"--oem 3 --psm {psm}").strip()
        raw_parts.append(f"{label}: {text}")
        result[label] = _clean_field_value(label, text)

    result["raw_text"] = "\n".join(raw_parts)
    return result


def extract_front(image_bytes: bytes) -> dict:
    img = _read_image(image_bytes)
    card = _locate_card(img)

    det_model = _get_detection_model()
    if det_model is not None:
        try:
            return _extract_fields_with_detector(card, det_model)
        except Exception:
            pass  # fall back to whole-card OCR + regex below

    prepped = _preprocess_for_ocr(card)
    raw_text = _ocr(prepped)
    fields = parse_ine_front(raw_text)
    return fields.to_dict()


# ---------------------------------------------------------------------------
# Back-side (MRZ) parsing — best effort, experimental
# ---------------------------------------------------------------------------

def _mrz_check_digit(data: str) -> int:
    weights = [7, 3, 1]
    values = {"<": 0}
    for i, ch in enumerate("0123456789"):
        values[ch] = i
    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        values[ch] = i + 10
    total = 0
    for i, ch in enumerate(data):
        total += values.get(ch, 0) * weights[i % 3]
    return total % 10


def _mrz_date(yymmdd: str) -> Optional[str]:
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if mm == 0 or mm > 12 or dd == 0 or dd > 31:
        return None
    # Heuristic pivot: treat as 20YY unless that would be in the future
    # relative to a birth-date-friendly range, otherwise 19YY.
    century = 2000 if yy <= 30 else 1900
    return f"{dd:02d}/{mm:02d}/{century + yy}"


def _extract_mrz_lines(raw_text: str) -> list:
    text_norm = _strip_accents_upper(raw_text)
    lines = []
    for line in text_norm.splitlines():
        candidate = re.sub(r"[^A-Z0-9<]", "", line)
        if len(candidate) >= 24:
            lines.append(candidate.ljust(30, "<")[:30])
    return lines[-3:] if len(lines) >= 3 else lines


def parse_mrz(raw_text: str) -> dict:
    """Best-effort parse of the ICAO 9303 TD1 machine-readable zone on the
    back of newer INE credentials. Returned as supplementary/cross-check
    data — always verify manually, OCR of the MRZ font is not guaranteed.
    """
    lines = _extract_mrz_lines(raw_text)
    result = {
        "mrz_detected": len(lines) == 3,
        "curp": None,
        "fecha_nacimiento": None,
        "fecha_vigencia": None,
        "sexo": None,
        "nombre": None,
        "raw_lines": lines,
        "raw_text": raw_text,
    }
    if len(lines) != 3:
        return result

    line1, line2, line3 = lines

    doc_number_field = line1[5:14].replace("<", "")
    if re.fullmatch(r"[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d", line1[15:30].replace("<", "")[:18]):
        result["curp"] = line1[15:30].replace("<", "")[:18]
    elif doc_number_field:
        result["clave_o_documento"] = doc_number_field

    result["fecha_nacimiento"] = _mrz_date(line2[0:6])
    sexo_char = line2[7]
    result["sexo"] = sexo_char if sexo_char in ("M", "F") else None
    result["fecha_vigencia"] = _mrz_date(line2[8:14])

    names_part = line3.rstrip("<")
    if "<<" in names_part:
        surnames, given = names_part.split("<<", 1)
        surnames = surnames.replace("<", " ").strip()
        given = given.replace("<", " ").strip()
        result["nombre"] = f"{given} {surnames}".strip() or None
    elif names_part:
        result["nombre"] = names_part.replace("<", " ").strip() or None

    return result


def extract_back(image_bytes: bytes) -> dict:
    img = _read_image(image_bytes)
    card = _locate_card(img)
    prepped = _preprocess_for_ocr(card)
    raw_text = _ocr(prepped, config=MRZ_TESS_CONFIG)
    mrz = parse_mrz(raw_text)
    if not mrz["mrz_detected"]:
        # Fall back to a normal OCR pass in case the back also carries
        # plain-text fields (some older INE formats do).
        raw_text_plain = _ocr(prepped)
        mrz["raw_text"] = raw_text_plain
    return mrz
