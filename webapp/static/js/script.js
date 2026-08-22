(() => {
  const files = { front: null, back: null };

  const dropzones = {
    front: document.getElementById("dropzone-front"),
    back: document.getElementById("dropzone-back"),
  };
  const inputs = {
    front: document.getElementById("input-front"),
    back: document.getElementById("input-back"),
  };
  const previews = {
    front: document.getElementById("preview-front"),
    back: document.getElementById("preview-back"),
  };

  const extractBtn = document.getElementById("extract-btn");
  const statusMsg = document.getElementById("status-msg");
  const resultsSection = document.getElementById("results");
  const resultsForm = document.getElementById("results-form");
  const mrzPanel = document.getElementById("mrz-panel");
  const mrzList = document.getElementById("mrz-list");
  const rawTextEl = document.getElementById("raw-text");

  let lastPayload = null;

  function setStatus(text, kind) {
    statusMsg.textContent = text || "";
    statusMsg.className = "status-msg" + (kind ? " " + kind : "");
  }

  function updateExtractAvailability() {
    extractBtn.disabled = !files.front;
  }

  function handleFile(side, file) {
    if (!file || !file.type.startsWith("image/")) {
      setStatus("Selecciona un archivo de imagen válido.", "error");
      return;
    }
    files[side] = file;
    const url = URL.createObjectURL(file);
    previews[side].src = url;
    previews[side].hidden = false;
    dropzones[side].querySelector(".dropzone-empty").hidden = true;
    dropzones[side].querySelector(".remove-btn").hidden = false;
    updateExtractAvailability();
  }

  function clearFile(side) {
    files[side] = null;
    previews[side].src = "";
    previews[side].hidden = true;
    dropzones[side].querySelector(".dropzone-empty").hidden = false;
    dropzones[side].querySelector(".remove-btn").hidden = true;
    inputs[side].value = "";
    updateExtractAvailability();
  }

  Object.entries(dropzones).forEach(([side, zone]) => {
    zone.addEventListener("click", (e) => {
      if (e.target.closest(".remove-btn")) return;
      inputs[side].click();
    });
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      handleFile(side, file);
    });
    zone.querySelector(".remove-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile(side);
    });
  });

  inputs.front.addEventListener("change", (e) => handleFile("front", e.target.files[0]));
  inputs.back.addEventListener("change", (e) => handleFile("back", e.target.files[0]));

  const MRZ_LABELS = {
    curp: "CURP",
    clave_o_documento: "Número de documento",
    fecha_nacimiento: "Fecha de nacimiento",
    fecha_vigencia: "Vigencia",
    sexo: "Sexo",
    nombre: "Nombre",
  };

  function renderMrz(back) {
    if (!back || back.error) {
      mrzPanel.hidden = true;
      return;
    }
    mrzList.innerHTML = "";
    let any = false;
    Object.entries(MRZ_LABELS).forEach(([key, label]) => {
      if (back[key]) {
        any = true;
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = back[key];
        mrzList.appendChild(dt);
        mrzList.appendChild(dd);
      }
    });
    if (!back.mrz_detected) {
      const note = document.createElement("p");
      note.className = "mrz-hint";
      note.textContent = "No se detectó una zona MRZ clara en el reverso; se muestra el texto OCR crudo abajo.";
      mrzPanel.appendChild(note);
    }
    mrzPanel.hidden = !any && back.mrz_detected;
    mrzPanel.hidden = false;
  }

  function currentFormValues() {
    const data = {};
    new FormData(resultsForm).forEach((value, key) => (data[key] = value));
    return data;
  }

  document.getElementById("copy-btn").addEventListener("click", async () => {
    const payload = { ...currentFormValues() };
    if (lastPayload && lastPayload.back) payload.reverso = lastPayload.back;
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setStatus("JSON copiado al portapapeles.", "ok");
    } catch {
      setStatus("No se pudo copiar automáticamente. Copia el texto manualmente.", "error");
    }
  });

  document.getElementById("download-btn").addEventListener("click", () => {
    const payload = { ...currentFormValues() };
    if (lastPayload && lastPayload.back) payload.reverso = lastPayload.back;
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ine_datos.json";
    a.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById("raw-toggle-btn").addEventListener("click", () => {
    rawTextEl.hidden = !rawTextEl.hidden;
  });

  extractBtn.addEventListener("click", async () => {
    if (!files.front) return;
    extractBtn.disabled = true;
    setStatus("Procesando imagen(es)…");
    resultsSection.hidden = true;

    const formData = new FormData();
    formData.append("front", files.front);
    if (files.back) formData.append("back", files.back);

    try {
      const res = await fetch("/api/extract", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || "Ocurrió un error al procesar la imagen.", "error");
        return;
      }

      lastPayload = data;
      const front = data.front || {};
      Object.keys(currentFormValues()).forEach((key) => {
        const input = resultsForm.elements[key];
        if (input) input.value = front[key] || "";
      });

      rawTextEl.textContent = [
        "--- Texto OCR (frente) ---",
        front.raw_text || "(sin texto)",
        data.back ? "\n--- Texto OCR (reverso) ---" : "",
        data.back ? data.back.raw_text || "(sin texto)" : "",
      ].join("\n");

      renderMrz(data.back);
      resultsSection.hidden = false;
      setStatus("Extracción completada. Revisa y corrige los campos si es necesario.", "ok");
    } catch (err) {
      setStatus("No se pudo conectar con el servidor: " + err.message, "error");
    } finally {
      extractBtn.disabled = !files.front;
    }
  });
})();
