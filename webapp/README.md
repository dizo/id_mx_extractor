# Extractor de datos INE (interfaz web)

Aplicación web para extraer automáticamente los datos impresos en una
credencial para votar (INE) mexicana a partir de una foto, usando
preprocesamiento de imagen (OpenCV) y OCR (Tesseract).

No depende de ningún modelo entrenado del resto del repositorio (no hay
ninguno versionado); usa detección de bordes/perspectiva clásica para
localizar la tarjeta y expresiones regulares ancladas a las etiquetas del
formato INE para extraer cada campo. Los resultados se muestran en un
formulario editable porque el OCR nunca es 100% exacto.

## Campos extraídos

- Nombre completo, domicilio
- CURP y clave de elector
- Fecha de nacimiento, sexo
- Año de registro, estado, municipio, localidad, sección
- Año de emisión y vigencia
- (Opcional) datos del reverso vía zona MRZ: CURP/documento, fecha de
  nacimiento, vigencia, sexo y nombre — funcionalidad experimental

## Requisitos

- Python 3.9+
- Tesseract OCR instalado en el sistema, con el paquete de idioma español:
  - Debian/Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-spa`
  - macOS (Homebrew): `brew install tesseract tesseract-lang`

## Instalación y ejecución

```bash
cd webapp
pip install -r requirements.txt
python app.py
```

Luego abre `http://localhost:5000` en el navegador.

## Uso

1. Sube la foto del frente de la INE (obligatorio).
2. Opcionalmente sube el reverso para intentar leer el código MRZ.
3. Haz clic en "Extraer datos".
4. Revisa y corrige los campos si el OCR se equivocó en algo.
5. Copia o descarga el resultado como JSON.

## Privacidad

Las imágenes se procesan en memoria para generar la respuesta y no se
guardan en disco ni en ninguna base de datos. Al tratarse de datos
personales sensibles (nombre, CURP, domicilio, fecha de nacimiento), usa
esta herramienta únicamente con tu propia identificación o con
autorización explícita del titular de la credencial.
