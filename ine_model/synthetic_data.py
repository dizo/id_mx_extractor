"""Synthetic Mexican INE dataset generator.

Produces (image, card-segmentation mask, per-field bounding boxes) triples
by rendering fake data onto the schematic template in ``template.py`` and
compositing it onto procedurally generated backgrounds with random
perspective, rotation, lighting and noise — the same overall recipe the
MIDV2020-based notebooks in this repo use, adapted to run without any
external dataset.

Nothing here is derived from real people or real INE images.
"""
from __future__ import annotations

import json
import os
import random
import string
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from .template import (
    CARD_H,
    CARD_W,
    FIELD_BOXES_FRAC,
    FIELD_NAMES,
    LABELS_ES,
    draw_blank_template,
    field_boxes_px,
)

_faker = Faker("es_MX")

_STATE_CODES = {
    "AGUASCALIENTES": "AS", "BAJA CALIFORNIA": "BC", "BAJA CALIFORNIA SUR": "BS",
    "CAMPECHE": "CC", "COAHUILA": "CL", "COLIMA": "CM", "CHIAPAS": "CS",
    "CHIHUAHUA": "CH", "CIUDAD DE MEXICO": "DF", "DURANGO": "DG",
    "GUANAJUATO": "GT", "GUERRERO": "GR", "HIDALGO": "HG", "JALISCO": "JC",
    "MEXICO": "MC", "MICHOACAN": "MN", "MORELOS": "MS", "NAYARIT": "NT",
    "NUEVO LEON": "NL", "OAXACA": "OC", "PUEBLA": "PL", "QUERETARO": "QT",
    "QUINTANA ROO": "QR", "SAN LUIS POTOSI": "SP", "SINALOA": "SL",
    "SONORA": "SR", "TABASCO": "TC", "TAMAULIPAS": "TS", "TLAXCALA": "TL",
    "VERACRUZ": "VZ", "YUCATAN": "YN", "ZACATECAS": "ZS",
}
_STATE_NAMES = list(_STATE_CODES.keys())

_CURP_CHARSET = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
_VOWELS = "AEIOU"
_INAPPROPRIATE_CURP_WORDS = {"BUEI", "BUEY", "CACA", "CACO", "CAGA", "CAGO", "CAKA", "CAKO",
                              "COGE", "COJA", "COJE", "COJI", "COJO", "CULO", "FETO", "GUEY",
                              "JOTO", "KACA", "KACO", "KAGA", "KAGO", "KOGE", "KOJO", "KAKA",
                              "KULO", "MAME", "MAMO", "MEAR", "MEAS", "MEON", "MION", "MOCO",
                              "MULA", "PEDA", "PEDO", "PENE", "PUTA", "PUTO", "QULO", "RATA",
                              "RUIN"}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _first_internal_vowel(word: str) -> str:
    for ch in word[1:]:
        if ch in _VOWELS:
            return ch
    return "X"


def _first_internal_consonant(word: str) -> str:
    for ch in word[1:]:
        if ch.isalpha() and ch not in _VOWELS:
            return ch
    return "X"


def _curp_char_value(ch: str) -> int:
    idx = _CURP_CHARSET.find(ch)
    return idx if idx >= 0 else 0


def generate_curp(paterno: str, materno: str, nombre: str, birthdate, sexo: str, estado_nombre: str) -> str:
    paterno_n = _strip_accents(paterno).upper() or "X"
    materno_n = _strip_accents(materno).upper() or "X"
    nombre_n = _strip_accents(nombre).upper() or "X"
    # Common one-word given names skip particles like "DE"/"DEL"/"LA" in Spanish
    # naming conventions; not handled here for simplicity (synthetic data only).

    prefix = (
        paterno_n[0]
        + _first_internal_vowel(paterno_n)
        + (materno_n[0] if materno_n else "X")
        + nombre_n[0]
    )
    if prefix in _INAPPROPRIATE_CURP_WORDS:
        prefix = prefix[:3] + "X"

    yy = f"{birthdate.year % 100:02d}"
    mm = f"{birthdate.month:02d}"
    dd = f"{birthdate.day:02d}"

    state_code = _STATE_CODES.get(estado_nombre, "NE")
    consonants = (
        _first_internal_consonant(paterno_n)
        + _first_internal_consonant(materno_n)
        + _first_internal_consonant(nombre_n)
    )
    differentiator = str(random.randint(0, 9))

    body17 = prefix + yy + mm + dd + sexo + state_code + consonants + differentiator
    total = sum(_curp_char_value(ch) * (18 - i) for i, ch in enumerate(body17))
    check_digit = (10 - (total % 10)) % 10

    return body17 + str(check_digit)


def generate_clave_elector(paterno: str, materno: str, nombre: str, birthdate, estado_num: int) -> str:
    letters_pool = [c for c in _strip_accents(paterno + materno + nombre).upper() if c.isalpha()]
    while len(letters_pool) < 6:
        letters_pool.append(random.choice(string.ascii_uppercase))
    letters = "".join(letters_pool[:6])
    yy = f"{birthdate.year % 100:02d}"
    return (
        letters
        + f"{estado_num:02d}"
        + f"{random.randint(1, 999):03d}"
        + yy
        + f"{random.randint(0, 999):03d}"
        + f"{random.randint(0, 99):02d}"
    )


@dataclass
class IneRecord:
    paterno: str
    materno: str
    nombre: str
    domicilio: str
    curp: str
    clave_elector: str
    fecha_nacimiento: str
    sexo: str
    anio_registro: str
    estado: str
    estado_num: int
    municipio: str
    localidad: str
    seccion: str
    emision: str
    vigencia: str

    def field_text(self) -> Dict[str, str]:
        return {
            "nombre": f"{self.paterno} {self.materno} {self.nombre}",
            "domicilio": self.domicilio,
            "clave_elector": self.clave_elector,
            "curp": self.curp,
            "fecha_nacimiento": self.fecha_nacimiento,
            "sexo": self.sexo,
            "anio_registro": self.anio_registro,
            "estado": self.estado,
            "municipio": self.municipio,
            "localidad": self.localidad,
            "seccion": self.seccion,
            "emision": self.emision,
            "vigencia": self.vigencia,
        }


def generate_fake_record() -> IneRecord:
    sexo = random.choice(["H", "M"])
    first_name = _faker.first_name_male() if sexo == "H" else _faker.first_name_female()
    paterno = _faker.last_name()
    materno = _faker.last_name()

    birthdate = _faker.date_of_birth(minimum_age=18, maximum_age=88)
    estado_nombre = random.choice(_STATE_NAMES)
    estado_num = _STATE_NAMES.index(estado_nombre) + 1

    curp = generate_curp(paterno, materno, first_name, birthdate, sexo, estado_nombre)
    clave_elector = generate_clave_elector(paterno, materno, first_name, birthdate, estado_num)

    min_registro_year = birthdate.year + 18
    anio_registro = random.randint(min_registro_year, min(min_registro_year + 40, 2024))
    emision = random.randint(anio_registro, min(anio_registro + 15, 2024))
    vigencia = f"{emision}-{emision + 10}"

    domicilio = f"{_faker.street_address()} COL {_faker.city_suffix()} {_faker.postcode()} {_faker.city()}".upper()

    return IneRecord(
        paterno=paterno.upper(),
        materno=materno.upper(),
        nombre=first_name.upper(),
        domicilio=domicilio,
        curp=curp,
        clave_elector=clave_elector,
        fecha_nacimiento=birthdate.strftime("%d/%m/%Y"),
        sexo=sexo,
        anio_registro=str(anio_registro),
        estado=estado_nombre,
        estado_num=estado_num,
        municipio=_faker.city().upper(),
        localidad=f"{random.randint(1, 9999):04d}",
        seccion=f"{random.randint(1, 6000):04d}",
        emision=str(emision),
        vigencia=vigencia,
    )


# ---------------------------------------------------------------------------
# Rendering fake data onto the card template
# ---------------------------------------------------------------------------

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.getlength(candidate) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_field_text(draw: ImageDraw.ImageDraw, text: str, box: Tuple[int, int, int, int], multiline: bool):
    x, y, w, h = box
    size = 17 if multiline else 18
    min_size = 9
    while size >= min_size:
        font = ImageFont.truetype(_FONT_PATH, size)
        line_h = int(size * 1.25)
        max_lines = max(1, h // line_h) if multiline else 1
        lines = _wrap_text(text, font, w) if multiline else [text]
        if not multiline and font.getlength(text) > w:
            size -= 1
            continue
        if len(lines) <= max_lines and all(font.getlength(line) <= w for line in lines):
            break
        size -= 1
    else:
        font = ImageFont.truetype(_FONT_PATH, min_size)
        line_h = int(min_size * 1.25)
        lines = _wrap_text(text, font, w)[: max(1, h // line_h)]

    ty = y
    for line in lines:
        draw.text((x, ty), line, font=font, fill=(20, 20, 25))
        ty += line_h


def render_card(record: IneRecord) -> np.ndarray:
    """Returns a BGR uint8 numpy array of the CARD_W x CARD_H rendered card."""
    template = draw_blank_template()
    img = template.image.copy()
    draw = ImageDraw.Draw(img)

    boxes = field_boxes_px()
    texts = record.field_text()
    multiline_fields = {"nombre", "domicilio"}

    for name, box in boxes.items():
        text = texts.get(name, "")
        if not text:
            continue
        _draw_field_text(draw, text, box, multiline=name in multiline_fields)

    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Scene compositing: random background + perspective warp + noise
# ---------------------------------------------------------------------------

def _random_background(w: int, h: int) -> np.ndarray:
    style = random.choice(["solid_noise", "gradient", "gradient_noise"])
    base_color = np.array([random.randint(40, 220) for _ in range(3)], dtype=np.float32)

    if style == "solid_noise":
        bg = np.tile(base_color, (h, w, 1))
        noise = np.random.normal(0, 18, (h, w, 3))
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    else:
        other_color = np.array([random.randint(20, 220) for _ in range(3)], dtype=np.float32)
        axis = random.choice(["horizontal", "vertical", "diagonal"])
        if axis == "horizontal":
            ramp = np.linspace(0, 1, w).reshape(1, w, 1)
            ramp = np.repeat(ramp, h, axis=0)
        elif axis == "vertical":
            ramp = np.linspace(0, 1, h).reshape(h, 1, 1)
            ramp = np.repeat(ramp, w, axis=1)
        else:
            xv, yv = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
            ramp = ((xv + yv) / 2)[:, :, None]
        bg = base_color * (1 - ramp) + other_color * ramp
        if style == "gradient_noise":
            bg += np.random.normal(0, 12, (h, w, 3))
        bg = np.clip(bg, 0, 255).astype(np.uint8)

    # A handful of random clutter shapes so the model doesn't assume a plain
    # background.
    for _ in range(random.randint(0, 4)):
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        shape = random.choice(["rect", "circle", "line"])
        pt1 = (random.randint(0, w), random.randint(0, h))
        pt2 = (random.randint(0, w), random.randint(0, h))
        if shape == "rect":
            cv2.rectangle(bg, pt1, pt2, color, -1)
        elif shape == "circle":
            cv2.circle(bg, pt1, random.randint(10, min(w, h) // 4), color, -1)
        else:
            cv2.line(bg, pt1, pt2, color, random.randint(1, 6))

    return bg


def _perspective_points(out_w: int, out_h: int) -> Tuple[np.ndarray, float]:
    scale = random.uniform(0.55, 0.92)
    target_w = scale * out_w
    target_h = target_w * CARD_H / CARD_W
    if target_h > out_h * 0.92:
        target_h = out_h * 0.92
        target_w = target_h * CARD_W / CARD_H

    cx = random.uniform(target_w / 2, out_w - target_w / 2)
    cy = random.uniform(target_h / 2, out_h - target_h / 2)

    corners = np.array(
        [
            [cx - target_w / 2, cy - target_h / 2],
            [cx + target_w / 2, cy - target_h / 2],
            [cx + target_w / 2, cy + target_h / 2],
            [cx - target_w / 2, cy + target_h / 2],
        ],
        dtype=np.float32,
    )

    angle = np.deg2rad(random.uniform(-20, 20))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = corners.mean(axis=0)
    corners = (corners - center) @ rot.T + center

    jitter = target_w * 0.05
    corners += np.random.uniform(-jitter, jitter, corners.shape)

    return corners.astype(np.float32), scale


def compose_scene(
    card_bgr: np.ndarray,
    out_size: Tuple[int, int] = (900, 700),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int, int, int]]]:
    """Composites a rendered card onto a random background.

    Returns (scene_bgr, mask, field_boxes_scene) where field_boxes_scene maps
    field name -> (xmin, ymin, xmax, ymax) in scene pixel coordinates,
    clipped to the frame, only for fields that stay at least 30% visible.
    """
    out_w, out_h = out_size
    background = _random_background(out_w, out_h)

    src_corners = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]], dtype=np.float32)
    dst_corners, _ = _perspective_points(out_w, out_h)
    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)

    warped_card = cv2.warpPerspective(card_bgr, matrix, (out_w, out_h))
    card_alpha = np.full((CARD_H, CARD_W), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(card_alpha, matrix, (out_w, out_h))

    scene = background.copy()
    mask_bool = warped_mask > 0
    scene[mask_bool] = warped_card[mask_bool]

    field_boxes_scene: Dict[str, Tuple[int, int, int, int]] = {}
    for name, (x, y, w, h) in field_boxes_px().items():
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        corners = corners.reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        xmin, ymin = warped.min(axis=0)
        xmax, ymax = warped.max(axis=0)
        orig_area = w * h
        cxmin, cymin = max(0, xmin), max(0, ymin)
        cxmax, cymax = min(out_w, xmax), min(out_h, ymax)
        clipped_area = max(0, cxmax - cxmin) * max(0, cymax - cymin)
        if orig_area <= 0 or clipped_area / orig_area < 0.3:
            continue
        field_boxes_scene[name] = (int(cxmin), int(cymin), int(cxmax), int(cymax))

    scene = _apply_photo_effects(scene)
    return scene, warped_mask, field_boxes_scene


def _apply_photo_effects(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)

    brightness = random.uniform(-30, 30)
    contrast = random.uniform(0.8, 1.25)
    out = (out - 127.5) * contrast + 127.5 + brightness

    if random.random() < 0.5:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    noise_sigma = random.uniform(0, 10)
    if noise_sigma > 0.5:
        out += np.random.normal(0, noise_sigma, out.shape)

    out = np.clip(out, 0, 255).astype(np.uint8)

    if random.random() < 0.6:
        quality = random.randint(45, 90)
        ok, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return out


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int,
    out_dir: str,
    seed: Optional[int] = None,
    out_size: Tuple[int, int] = (900, 700),
) -> str:
    """Generates ``n`` synthetic samples under ``out_dir``:

    out_dir/
      images/00000.jpg ...
      masks/00000.png ...
      annotations.json  -- list of {"image", "mask", "boxes": [{"label","bbox"}]}

    Returns the path to annotations.json.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    annotations = []
    for i in range(n):
        record = generate_fake_record()
        card_bgr = render_card(record)
        scene, mask, boxes = compose_scene(card_bgr, out_size=out_size)

        image_name = f"{i:05d}.jpg"
        mask_name = f"{i:05d}.png"
        cv2.imwrite(os.path.join(images_dir, image_name), scene)
        cv2.imwrite(os.path.join(masks_dir, mask_name), mask)

        annotations.append(
            {
                "image": image_name,
                "mask": mask_name,
                "boxes": [
                    {"label": label, "bbox": list(bbox)} for label, bbox in boxes.items()
                ],
            }
        )

    ann_path = os.path.join(out_dir, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as fh:
        json.dump(annotations, fh, ensure_ascii=False, indent=2)

    return ann_path
