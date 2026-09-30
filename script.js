(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var drop = $("drop"), fileInput = $("fileInput");
  var dropEmpty = $("dropEmpty"), dropPreview = $("dropPreview");
  var previewImg = $("previewImg"), fileName = $("fileName"), fileSize = $("fileSize");
  var removeBtn = $("removeBtn"), analyzeBtn = $("analyzeBtn");
  var errorBox = $("error"), loading = $("loading");
  var result = $("result"), resetBtn = $("resetBtn");

  var selectedFile = null;
  var previewUrl = null;

  var MSG = {
    none: "Please select a chest X-ray image.",
    type: "Only JPG, JPEG and PNG images are supported.",
    empty: "The selected file is empty.",
    backend: "Unable to analyze the image.\nPlease make sure the FastAPI server is running.",
    network: "Could not connect to the prediction server."
  };

  /* ---------- helpers ---------- */
  function showError(text) { errorBox.textContent = text; errorBox.hidden = false; }
  function clearError() { errorBox.hidden = true; errorBox.textContent = ""; }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function pct(value) { return (Number(value) * 100).toFixed(2) + "%"; }

  function isAllowedType(file) {
    var okMime = file.type === "image/jpeg" || file.type === "image/png";
    var okExt = /\.(jpe?g|png)$/i.test(file.name);
    return okMime && okExt;
  }

  /* ---------- file selection ---------- */
  function setFile(file) {
    clearError();
    hideResult();
    if (!file) return;
    if (!isAllowedType(file)) { showError(MSG.type); return; }
    if (file.size === 0) { showError(MSG.empty); return; }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    dropEmpty.hidden = true;
    dropPreview.hidden = false;
  }

  function clearFile() {
    selectedFile = null;
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    previewImg.removeAttribute("src");
    fileInput.value = "";
    dropPreview.hidden = true;
    dropEmpty.hidden = false;
  }

  drop.addEventListener("click", function (e) {
    if (e.target.closest("#removeBtn")) return;
    fileInput.click();
  });
  drop.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", function () { setFile(fileInput.files[0]); });

  ["dragenter", "dragover"].forEach(function (evt) {
    drop.addEventListener(evt, function (e) { e.preventDefault(); drop.classList.add("dragover"); });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    drop.addEventListener(evt, function (e) { e.preventDefault(); drop.classList.remove("dragover"); });
  });
  drop.addEventListener("drop", function (e) {
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) setFile(files[0]);
  });
  // Stop the browser from opening a file dropped outside the box
  window.addEventListener("dragover", function (e) { e.preventDefault(); });
  window.addEventListener("drop", function (e) { e.preventDefault(); });

  removeBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    clearError();
    clearFile();
  });

  /* ---------- prediction ---------- */
  function setLoading(on) {
    loading.hidden = !on;
    analyzeBtn.disabled = on;
  }

  function hideResult() {
    result.hidden = true;
    result.classList.remove("normal", "pneumonia");
  }

  analyzeBtn.addEventListener("click", function () {
    clearError();
    hideResult();
    if (!selectedFile) { showError(MSG.none); return; }
    if (selectedFile.size === 0) { showError(MSG.empty); return; }

    var formData = new FormData();
    formData.append("file", selectedFile);
    setLoading(true);

    fetch("/predict", { method: "POST", body: formData })
      .then(function (res) {
        if (!res.ok) throw new Error("backend");
        return res.json();
      })
      .then(function (data) {
        if (!data || (data.prediction !== "NORMAL" && data.prediction !== "PNEUMONIA") ||
            typeof data.probability !== "number" || typeof data.confidence !== "number") {
          throw new Error("backend");
        }
        showResult(data);
      })
      .catch(function (err) {
        // fetch rejects with TypeError when the server can't be reached
        showError(err instanceof TypeError ? MSG.network : MSG.backend);
      })
      .then(function () { setLoading(false); });
  });

  function showResult(data) {
    var isPneumonia = data.prediction === "PNEUMONIA";
    var prob = Math.max(0, Math.min(100, data.probability * 100));

    result.classList.remove("normal", "pneumonia");
    result.classList.add(isPneumonia ? "pneumonia" : "normal");
    $("verdict").textContent = isPneumonia ? "! PNEUMONIA" : "✓ NORMAL";
    $("predText").textContent = isPneumonia ? "Pneumonia detected" : "Normal";
    $("confText").textContent = pct(data.confidence);
    $("probText").textContent = pct(data.probability);
    $("barValue").textContent = pct(data.probability);
    $("bar").setAttribute("aria-valuenow", prob.toFixed(2));

    $("barFill").style.width = "0";
    result.hidden = false;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { $("barFill").style.width = prob + "%"; });
    });
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ---------- reset ---------- */
  resetBtn.addEventListener("click", function () {
    hideResult();
    clearError();
    clearFile();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  /* ---------- server status ---------- */
  fetch("/health")
    .then(function (res) { return res.ok ? res.json() : Promise.reject(); })
    .then(function (d) {
      if (!d || d.model_loaded !== true) throw new Error();
    })
    .catch(function () {
      $("status").classList.add("offline");
      $("statusText").textContent = "Server not reachable";
    });
})();

/* ---------- lung breathing: WebGL warp of the lung render ---------- */
(function () {
  "use strict";
  var img = document.getElementById("lungImg");
  var canvas = document.getElementById("lungCanvas");
  var phaseEl = document.getElementById("phase");
  var PERIOD = 5000; // one full breath: ~2.5s in, ~2.5s out

  function fallback() {
    canvas.hidden = true;
    img.hidden = false;
    img.classList.add("css-breath");
  }

  function start() {
    var gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) { fallback(); return; }

    var vs = "attribute vec2 p; varying vec2 v;" +
      "void main(){ v = vec2(p.x*0.5+0.5, 0.5-p.y*0.5); gl_Position = vec4(p,0.0,1.0); }";
    var fs = "precision mediump float; varying vec2 v; uniform sampler2D tex; uniform float b;" +
      "void main(){" +
      "  vec2 c = vec2(0.583, 0.58);" +
      "  vec2 d = (v - c) / vec2(0.22, 0.30);" +
      "  float f = exp(-dot(d, d));" +
      "  vec2 uv = v - (v - c) * 0.11 * b * f;" +                        // lungs widen
      "  uv.y -= 0.014 * b * f * smoothstep(0.0, 0.3, v.y - c.y);" +       // diaphragm drops
      "  uv.y += 0.006 * b * (1.0 - smoothstep(0.0, 0.45, v.y));" +        // shoulders lift
      "  vec4 col = texture2D(tex, uv);" +
      "  float lum = dot(col.rgb, vec3(0.3, 0.5, 0.2));" +
      "  col.rgb *= 1.0 + 0.4 * b * f;" +                                  // lungs brighten
      "  col.rgb += vec3(0.0, 0.15, 0.35) * lum * b * f * 0.6;" +
      "  gl_FragColor = col; }";

    function compile(type, src) {
      var sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      return gl.getShaderParameter(sh, gl.COMPILE_STATUS) ? sh : null;
    }
    var vsh = compile(gl.VERTEX_SHADER, vs), fsh = compile(gl.FRAGMENT_SHADER, fs);
    if (!vsh || !fsh) { fallback(); return; }
    var prog = gl.createProgram();
    gl.attachShader(prog, vsh);
    gl.attachShader(prog, fsh);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { fallback(); return; }
    gl.useProgram(prog);

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);

    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    } catch (e) { fallback(); return; }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    var bLoc = gl.getUniformLocation(prog, "b");
    var lastPhase = "";

    function frame(now) {
      var t = (now % PERIOD) / PERIOD;                 // 0..1 through one breath
      var b = 0.5 - 0.5 * Math.cos(t * 2 * Math.PI);   // smooth 0 -> 1 -> 0
      gl.uniform1f(bLoc, b);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      var ph = t < 0.5 ? "Inhale" : "Exhale";
      if (ph !== lastPhase) { phaseEl.textContent = ph; lastPhase = ph; }
      requestAnimationFrame(frame);
    }

    canvas.hidden = false;
    img.hidden = true;
    requestAnimationFrame(frame);
  }

  if (img.complete && img.naturalWidth) start();
  else img.addEventListener("load", start);
})();
