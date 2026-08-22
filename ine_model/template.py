"""Schematic (non-official) INE-front card template used to synthesize
training data.

This intentionally does NOT reproduce the real INE's official artwork,
security holograms, or exact colors/typography — it only mimics the general
layout (photo on the left, stacked text fields on the right, a small
state/municipality/section table, a bottom validity band) closely enough
that a model trained on it transfers reasonably well to real photos. Field
positions are approximate; tune ``FIELD_BOXES`` against a few real reference
photos (your own INE, with consent) if you need tighter crops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

# ID-1 card proportions (85.6mm x 54mm) rendered at ~12px/mm.
CARD_W = 1011
CARD_H = 638

_FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = _FONT_PATH_BOLD if bold else _FONT_PATH_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


# Fields expressed as fractions of (CARD_W, CARD_H): (x, y, w, h).
# The *_label boxes are where the small field caption is drawn; the field
# itself (used as detection ground truth) is the value box below/right of it.
FIELD_BOXES_FRAC: Dict[str, Tuple[float, float, float, float]] = {
    "nombre": (0.335, 0.145, 0.630, 0.115),
    "domicilio": (0.335, 0.275, 0.630, 0.095),
    "clave_elector": (0.335, 0.385, 0.630, 0.045),
    "curp": (0.335, 0.435, 0.400, 0.045),
    "fecha_nacimiento": (0.335, 0.485, 0.230, 0.045),
    "sexo": (0.575, 0.485, 0.090, 0.045),
    "anio_registro": (0.680, 0.485, 0.180, 0.045),
    "estado": (0.030, 0.660, 0.150, 0.075),
    "municipio": (0.190, 0.660, 0.150, 0.075),
    "seccion": (0.350, 0.660, 0.130, 0.075),
    "localidad": (0.490, 0.660, 0.180, 0.075),
    "emision": (0.680, 0.660, 0.140, 0.075),
    "vigencia": (0.830, 0.660, 0.150, 0.075),
}

# Non-field decorative regions, only used when drawing the template.
_PHOTO_BOX_FRAC = (0.030, 0.145, 0.270, 0.400)
_SIGNATURE_BOX_FRAC = (0.030, 0.560, 0.270, 0.060)
_HEADER_H_FRAC = 0.115
_HOLOGRAM_BOX_FRAC = (0.855, 0.780, 0.120, 0.170)

LABELS_ES = {
    "nombre": "NOMBRE",
    "domicilio": "DOMICILIO",
    "clave_elector": "CLAVE DE ELECTOR",
    "curp": "CURP",
    "fecha_nacimiento": "FECHA DE NACIMIENTO",
    "sexo": "SEXO",
    "anio_registro": "AÑO DE REGISTRO",
    "estado": "ESTADO",
    "municipio": "MUNICIPIO",
    "seccion": "SECCIÓN",
    "localidad": "LOCALIDAD",
    "emision": "EMISIÓN",
    "vigencia": "VIGENCIA",
}


def frac_to_px(box_frac: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x, y, w, h = box_frac
    return int(x * CARD_W), int(y * CARD_H), int(w * CARD_W), int(h * CARD_H)


def field_boxes_px() -> Dict[str, Tuple[int, int, int, int]]:
    return {name: frac_to_px(box) for name, box in FIELD_BOXES_FRAC.items()}


FIELD_NAMES = list(FIELD_BOXES_FRAC.keys())


@dataclass
class CardTemplate:
    image: Image.Image
    field_label_font: ImageFont.FreeTypeFont
    field_value_font: ImageFont.FreeTypeFont


def draw_blank_template(header_color=(0, 90, 80), accent_color=(0, 130, 110)) -> CardTemplate:
    """Draws an empty schematic INE-front layout (no field values yet)."""
    img = Image.new("RGB", (CARD_W, CARD_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    header_h = int(_HEADER_H_FRAC * CARD_H)
    draw.rectangle([0, 0, CARD_W, header_h], fill=header_color)
    draw.text(
        (16, header_h // 2 - 12),
        "INSTITUTO NACIONAL ELECTORAL",
        font=_font(20, bold=True),
        fill=(255, 255, 255),
    )
    draw.text(
        (16, header_h + 6),
        "CREDENCIAL PARA VOTAR",
        font=_font(14, bold=True),
        fill=(40, 40, 40),
    )

    # Photo placeholder.
    px, py, pw, ph = frac_to_px(_PHOTO_BOX_FRAC)
    draw.rectangle([px, py, px + pw, py + ph], outline=(120, 120, 120), width=2, fill=(225, 225, 228))
    draw.line([px, py, px + pw, py + ph], fill=(200, 200, 205), width=1)
    draw.line([px + pw, py, px, py + ph], fill=(200, 200, 205), width=1)
    # Generic head-and-shoulders silhouette so the photo box doesn't look empty.
    cx, cy = px + pw // 2, py + int(ph * 0.35)
    r = int(pw * 0.28)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(190, 190, 196))
    draw.pieslice(
        [px + int(pw * 0.1), py + int(ph * 0.55), px + int(pw * 0.9), py + int(ph * 1.3)],
        180,
        360,
        fill=(190, 190, 196),
    )

    # Signature placeholder.
    sx, sy, sw, sh = frac_to_px(_SIGNATURE_BOX_FRAC)
    draw.line([sx, sy + sh, sx + sw, sy + sh], fill=(120, 120, 120), width=1)
    draw.text((sx, sy), "FIRMA", font=_font(11), fill=(120, 120, 120))

    # Hologram / QR placeholder, bottom-right.
    hx, hy, hw, hh = frac_to_px(_HOLOGRAM_BOX_FRAC)
    draw.rectangle([hx, hy, hx + hw, hy + hh], fill=(235, 235, 240), outline=(180, 180, 185))
    for i in range(0, hw, 8):
        draw.line([hx + i, hy, hx + i, hy + hh], fill=(210, 210, 218))

    # Field captions.
    label_font = _font(11, bold=True)
    value_font = _font(16)
    for name, box_frac in FIELD_BOXES_FRAC.items():
        x, y, w, h = frac_to_px(box_frac)
        draw.text((x, y - 14), LABELS_ES[name], font=label_font, fill=accent_color)
        draw.line([x, y, x + w, y], fill=(210, 210, 215), width=1)

    draw.rectangle([0, 0, CARD_W - 1, CARD_H - 1], outline=(160, 160, 165), width=2)

    return CardTemplate(image=img, field_label_font=label_font, field_value_font=value_font)
