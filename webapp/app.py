"""Flask web app: upload an INE (Mexican voter ID) photo and get its fields
extracted via OCR. See ine_extractor.py for the extraction logic."""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from ine_extractor import ExtractionError, extract_back, extract_front

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB per request

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/extract")
def api_extract():
    front_file = request.files.get("front")
    back_file = request.files.get("back")

    if front_file is None or front_file.filename == "":
        return jsonify({"error": "Debes subir al menos la foto del frente de la INE."}), 400

    if not _allowed_file(front_file.filename):
        return jsonify({"error": "Formato de imagen no soportado. Usa JPG, PNG, WEBP o BMP."}), 400

    result = {}
    try:
        result["front"] = extract_front(front_file.read())
    except ExtractionError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001 - surface unexpected OCR/image errors to the UI
        return jsonify({"error": f"No se pudo procesar la imagen del frente: {exc}"}), 500

    if back_file is not None and back_file.filename != "":
        if not _allowed_file(back_file.filename):
            return jsonify({"error": "Formato de imagen no soportado para el reverso."}), 400
        try:
            result["back"] = extract_back(back_file.read())
        except ExtractionError as exc:
            result["back"] = {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            result["back"] = {"error": f"No se pudo procesar el reverso: {exc}"}

    return jsonify(result)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "La imagen es demasiado grande (máx. 12 MB)."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
