# Extractor de datos INE (interfaz web)

Aplicación web para extraer automáticamente los datos impresos en una
credencial para votar (INE) mexicana a partir de una foto, usando
preprocesamiento de imagen (OpenCV) y OCR (Tesseract).

Por defecto usa detección de bordes/perspectiva clásica (OpenCV) para
localizar la tarjeta y expresiones regulares ancladas a las etiquetas del
formato INE para extraer cada campo — no requiere ningún modelo entrenado.
Los resultados se muestran en un formulario editable porque el OCR nunca es
100% exacto.

Si entrenas los modelos de `notebooks/train_ine_model.ipynb` (segmentación
para localizar la tarjeta + detección de campos), `ine_extractor.py` los usa
automáticamente en cuanto coloques los pesos en `webapp/model/` — mejora
mucho la precisión porque hace OCR campo por campo en vez de sobre todo el
texto. Sin esos pesos, la app sigue funcionando igual con la heurística.
Ver `webapp/model/README.md`.

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

Para usar los modelos entrenados (opcional), instala también las
dependencias de inferencia y coloca los pesos entrenados:

```bash
pip install -r requirements-model.txt
# copia unet_ine.h5 y field_detector.pth a webapp/model/
```

## Docker

Desde la raíz del repositorio (el `Dockerfile` necesita tanto `webapp/` como
`ine_model/`):

```bash
docker compose up --build
```

Esto construye la imagen (Tesseract + español ya incluidos) y expone la app
en `http://localhost:5000`. `docker-compose.yml` monta `./webapp/model` como
volumen de solo lectura, así que si generas los pesos entrenados con el
notebook, colócalos ahí y el contenedor los recoge sin reconstruir la imagen.

Por defecto la imagen NO incluye TensorFlow/PyTorch (para mantenerla
liviana), así que aunque coloques los pesos seguirá usando la heurística a
menos que reconstruyas con:

```bash
docker compose build --build-arg INSTALL_MODEL_DEPS=true
docker compose up
```

Sin `docker compose`, equivalente con `docker` a secas:

```bash
docker build -t ine-extractor .
docker run -p 5000:5000 -v "$(pwd)/webapp/model:/app/webapp/model:ro" ine-extractor
```

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
