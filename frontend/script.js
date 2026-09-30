(function () {
  "use strict";

  // Backend base URL. Use "" instead if FastAPI serves this frontend itself.
  const API_URL = "https://farmerfriend-9fia.onrender.com";
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), fileInput = $("fileInput"), analyzeBtn = $("analyzeBtn");
  const errorBox = $("error"), result = $("result");
  let selectedFile = null, previewUrl = null, busy = false;

  const showError = (t) => { errorBox.textContent = t; errorBox.hidden = false; };
  const clearError = () => { errorBox.hidden = true; errorBox.textContent = ""; };
  const pct = (v) => (v * 100).toFixed(2) + "%";
  const fmtSize = (b) => b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : (b / 1048576).toFixed(2) + " MB";
  const allowed = (f) => (f.type === "image/jpeg" || f.type === "image/png") && /\.(jpe?g|png)$/i.test(f.name);

  function hideResult() { result.hidden = true; result.classList.remove("normal", "pneumonia"); }

  function setFile(file) {
    clearError(); hideResult();
    if (!file) return;
    if (!allowed(file)) return showError("Only JPG, JPEG and PNG images are supported.");
    if (file.size === 0) return showError("The selected file is empty.");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    $("previewImg").src = previewUrl;
    $("fileName").textContent = file.name;
    $("fileSize").textContent = fmtSize(file.size);
    $("dropEmpty").hidden = true; $("dropPreview").hidden = false;
    analyzeBtn.disabled = false;
  }

  function clearFile() {
    selectedFile = null;
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    $("previewImg").removeAttribute("src");
    fileInput.value = "";
    $("dropPreview").hidden = true; $("dropEmpty").hidden = false;
    analyzeBtn.disabled = true;
  }

  drop.addEventListener("click", (e) => { if (!e.target.closest("#removeBtn")) fileInput.click(); });
  drop.addEventListener("keydown", (e) => {
    if (e.target !== drop) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    if (e.key === "Escape" && selectedFile) { clearFile(); clearError(); hideResult(); }
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("dragover"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files; if (f.length) setFile(f[0]); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
  $("removeBtn").addEventListener("click", (e) => { e.stopPropagation(); clearError(); hideResult(); clearFile(); });

  function setBusy(on) {
    busy = on;
    analyzeBtn.disabled = on || !selectedFile;
    analyzeBtn.querySelector(".spinner").hidden = !on;
    $("analyzeLabel").textContent = on ? "Analyzing..." : "Analyze X-ray";
  }

  analyzeBtn.addEventListener("click", async () => {
    if (busy) return;
    clearError(); hideResult();
    if (!selectedFile) return showError("No X-ray selected.");
    if (selectedFile.size === 0) return showError("The selected file is empty.");
    const fd = new FormData();
    fd.append("file", selectedFile);
    setBusy(true);
    try {
      let res;
      try { res = await fetch(API_URL + "/predict", { method: "POST", body: fd }); }
      catch { showError("Could not connect to the prediction server."); setStatus(false); return; }
      if (!res.ok) return showError("Unable to analyze the image.");
      let data;
      try { data = await res.json(); } catch { data = null; }
      if (!data || !["NORMAL", "PNEUMONIA"].includes(data.prediction) ||
          typeof data.probability !== "number" || typeof data.confidence !== "number")
        return showError("The prediction server returned an unexpected response.");
      showResult(data);
    } finally { setBusy(false); }
  });

  function showResult(d) {
    const bad = d.prediction === "PNEUMONIA";
    result.classList.add(bad ? "pneumonia" : "normal");
    $("verdict").textContent = d.prediction;
    $("probText").textContent = pct(d.probability);
    $("confText").textContent = pct(d.confidence);
    $("explain").textContent = "Based on the model's classification, the uploaded image was classified as " + d.prediction + ".";
    const mark = $("mark");
    mark.style.left = "0%";
    result.hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() => { mark.style.left = Math.min(100, Math.max(0, d.probability * 100)) + "%"; }));
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  $("resetBtn").addEventListener("click", () => {
    hideResult(); clearError(); clearFile();
    $("analysis").scrollIntoView({ behavior: "smooth" });
  });

  /* server status */
  function setStatus(ok) {
    const s = $("status");
    s.classList.toggle("online", ok); s.classList.toggle("offline", !ok);
    $("statusText").textContent = ok ? "AI model online" : "Server offline";
    $("techStatus").textContent = ok ? "Live API connection" : "Offline";
  }
  async function checkHealth() {
    try {
      const r = await fetch(API_URL + "/health");
      const d = await r.json();
      setStatus(r.ok && d.status === "healthy" && d.model_loaded === true);
    } catch { setStatus(false); }
  }
  checkHealth(); setInterval(checkHealth, 15000);

  /* mobile nav */
  const nav = $("nav"), toggle = $("navToggle");
  toggle.addEventListener("click", () => toggle.setAttribute("aria-expanded", nav.classList.toggle("open")));
  nav.addEventListener("click", (e) => { if (e.target.tagName === "A") { nav.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); } });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { nav.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); } });

  /* lung breathing: one 5s cycle driven by a single clock so text and motion stay in sync */
  const lungs = $("lungs"), ribs = $("ribs"), dia = $("diaphragm"), phase = $("phase");
  const PERIOD = 5000;
  function apply(b) {
    lungs.style.transform = "scale(" + (1 + 0.07 * b) + "," + (1 + 0.045 * b) + ")";
    lungs.style.filter = "brightness(" + (0.93 + 0.14 * b) + ")";
    ribs.style.transform = "scale(" + (1 + 0.025 * b) + ")";
    dia.style.transform = "translateY(" + (7 * b) + "px)";
  }
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    apply(0.5); phase.textContent = "";
  } else {
    let last = "";
    (function frame(now) {
      const t = (now % PERIOD) / PERIOD;
      apply(0.5 - 0.5 * Math.cos(t * 2 * Math.PI));
      const p = t < 0.5 ? "Inhale" : "Exhale";
      if (p !== last) { phase.textContent = p; last = p; }
      requestAnimationFrame(frame);
    })(performance.now());
  }
})();
