/**
 * NeuroX: Clinical Epileptic Seizure Staging & Prediction Console
 * Client-Side Application Logic & 18-Channel Oscilloscope Canvas Engine
 */

// 18 Standard Channels
const STANDARD_CHANNELS = [
  "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
  "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
  "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
  "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
  "FZ-CZ", "CZ-PZ",
];

// Application State
const STATE = {
  presets: [],
  activePresetId: null,
  activePatientId: "chb01",
  waveformData: {},
  consecutiveWindows: 2,
  gain: 1.0,
  isSweeping: true,
  sweepX: 0,
  sweepSpeed: 2.2, // pixels per frame
  lastPrediction: null,
  customTelemetry: null,
};

// DOM Element Selectors
const DOM = {
  systemClock: document.getElementById("systemClock"),
  presetsGrid: document.getElementById("presetsGrid"),
  eegCanvas: document.getElementById("eegCanvas"),
  canvasTimestamp: document.getElementById("canvasTimestamp"),
  recordingSource: document.getElementById("recordingSource"),
  btnToggleSweep: document.getElementById("btnToggleSweep"),
  sweepIcon: document.getElementById("sweepIcon"),
  sweepText: document.getElementById("sweepText"),
  gainSlider: document.getElementById("gainSlider"),
  gainVal: document.getElementById("gainVal"),
  
  // Spectral band bars
  valDelta: document.getElementById("valDelta"),
  barDelta: document.getElementById("barDelta"),
  valTheta: document.getElementById("valTheta"),
  barTheta: document.getElementById("barTheta"),
  valAlpha: document.getElementById("valAlpha"),
  barAlpha: document.getElementById("barAlpha"),
  valBeta: document.getElementById("valBeta"),
  barBeta: document.getElementById("barBeta"),
  valGamma: document.getElementById("valGamma"),
  barGamma: document.getElementById("barGamma"),

  // Sliders
  sliderTB: document.getElementById("sliderTB"),
  valSliderTB: document.getElementById("valSliderTB"),
  sliderSync: document.getElementById("sliderSync"),
  valSliderSync: document.getElementById("valSliderSync"),
  sliderAsym: document.getElementById("sliderAsym"),
  valSliderAsym: document.getElementById("valSliderAsym"),
  sliderAmp: document.getElementById("sliderAmp"),
  valSliderAmp: document.getElementById("valSliderAmp"),
  btnResetSliders: document.getElementById("btnResetSliders"),
  btnPredict: document.getElementById("btnPredict"),
  actionStatus: document.getElementById("actionStatus"),

  // Staging Panel
  stagePanel: document.getElementById("stagePanel"),
  modelUsedBadge: document.getElementById("modelUsedBadge"),
  stageBadge: document.getElementById("stageBadge"),
  stageState: document.getElementById("stageState"),
  stageName: document.getElementById("stageName"),
  gaugeCircle: document.getElementById("gaugeCircle"),
  gaugeVal: document.getElementById("gaugeVal"),
  riskLevelPill: document.getElementById("riskLevelPill"),
  leadTimeVal: document.getElementById("leadTimeVal"),

  // Persistence
  persistenceCard: document.getElementById("persistenceCard"),
  persistenceIcon: document.getElementById("persistenceIcon"),
  alarmStateBadge: document.getElementById("alarmStateBadge"),
  persistenceDesc: document.getElementById("persistenceDesc"),
  step1: document.getElementById("step1"),
  step2: document.getElementById("step2"),

  // Consensus
  probPers: document.getElementById("probPers"),
  barPers: document.getElementById("barPers"),
  probRF: document.getElementById("probRF"),
  barRF: document.getElementById("barRF"),
  probCNN: document.getElementById("probCNN"),
  barCNN: document.getElementById("barCNN"),
  probLSTM: document.getElementById("probLSTM"),
  barLSTM: document.getElementById("barLSTM"),

  // Upload elements
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  btnUploadTrigger: document.getElementById("btnUploadTrigger"),
  btnBrowseFile: document.getElementById("btnBrowseFile"),
  uploadProgressWrap: document.getElementById("uploadProgressWrap"),
  progressBarFill: document.getElementById("progressBarFill"),
  progressText: document.getElementById("progressText"),
  progressPercent: document.getElementById("progressPercent"),
  uploadedBanner: document.getElementById("uploadedBanner"),
  uploadedFileName: document.getElementById("uploadedFileName"),
  uploadedMeta: document.getElementById("uploadedMeta"),
  btnClearUploaded: document.getElementById("btnClearUploaded"),

  // Protocol & Modal
  protocolList: document.getElementById("protocolList"),
  btnPrintReport: document.getElementById("btnPrintReport"),
  reportModal: document.getElementById("reportModal"),
  modalReportContent: document.getElementById("modalReportContent"),
  btnCloseModal: document.getElementById("btnCloseModal"),
  btnCloseModal2: document.getElementById("btnCloseModal2"),
  btnPrintWindow: document.getElementById("btnPrintWindow"),
};

// ==============================================================================
// Initialization
// ==============================================================================

window.addEventListener("DOMContentLoaded", async () => {
  initClock();
  initCanvas();
  bindEvents();
  await loadPresets();
});

function initClock() {
  function update() {
    const now = new Date();
    DOM.systemClock.textContent = now.toTimeString().split(" ")[0] + " UTC";
  }
  update();
  setInterval(update, 1000);
}

// ==============================================================================
// Presets Loading & Rendering
// ==============================================================================

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    const data = await res.json();
    STATE.presets = data.presets || [];
    renderPresets();
    
    // Auto-select first preictal alert preset (chb01_preictal or first available)
    if (STATE.presets.length > 1) {
      selectPreset(STATE.presets[1].id);
    } else if (STATE.presets.length > 0) {
      selectPreset(STATE.presets[0].id);
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
    DOM.presetsGrid.innerHTML = `
      <div class="preset-card loading-card">
        <span class="text-danger">Failed to connect to backend server. Make sure web/server.py is running.</span>
      </div>
    `;
  }
}

function renderPresets() {
  DOM.presetsGrid.innerHTML = "";
  STATE.presets.forEach((p) => {
    const card = document.createElement("div");
    card.className = `preset-card ${p.id === STATE.activePresetId ? "active" : ""}`;
    card.dataset.presetId = p.id;

    const isPreictal = p.stage_true === "PREICTAL";
    const badgeClass = isPreictal ? "badge-danger" : "badge-success";

    card.innerHTML = `
      <div class="preset-top">
        <span class="preset-patient">${p.patient_id.toUpperCase()}</span>
        <span class="badge ${badgeClass}">${p.stage_true}</span>
      </div>
      <div class="preset-title">${p.name}</div>
      <div class="preset-footer">
        <span>${p.edf_name}</span>
        <span>${isPreictal ? "Warning Lead Time" : "Baseline State"}</span>
      </div>
    `;

    card.addEventListener("click", () => selectPreset(p.id));
    DOM.presetsGrid.appendChild(card);
  });
}

async function selectPreset(presetId) {
  STATE.activePresetId = presetId;

  // Highlight card
  document.querySelectorAll(".preset-card").forEach((c) => {
    c.classList.toggle("active", c.dataset.presetId === presetId);
  });

  DOM.actionStatus.textContent = "Loading clinical EEG telemetry...";

  try {
    const res = await fetch(`/api/preset/${presetId}`);
    const preset = await res.json();

    STATE.activePatientId = preset.patient_id;
    STATE.waveformData = preset.waveform_data || {};
    STATE.customTelemetry = preset.global_telemetry;

    // Update metadata tags
    DOM.recordingSource.textContent = `${preset.edf_name} • Win #${preset.window_index}`;
    DOM.canvasTimestamp.textContent = `Window: [${preset.start_sec.toFixed(1)}s – ${preset.end_sec.toFixed(1)}s] • 18-CH Longitudinal Bipolar`;

    // Update Spectral Bars
    updateSpectralBars(preset.global_telemetry);

    // Update Sliders
    DOM.sliderTB.value = preset.global_telemetry.theta_beta_ratio || 1.89;
    DOM.valSliderTB.textContent = DOM.sliderTB.value;

    DOM.sliderSync.value = preset.global_telemetry.global_synchrony || 0.45;
    DOM.valSliderSync.textContent = DOM.sliderSync.value;

    DOM.sliderAsym.value = preset.global_telemetry.hemispheric_asymmetry || 1.05;
    DOM.valSliderAsym.textContent = DOM.sliderAsym.value;

    DOM.sliderAmp.value = 1.0;
    DOM.valSliderAmp.textContent = "1.00x";

    // Reset sweep bar to left
    STATE.sweepX = 0;

    // Run prediction
    await executePrediction(presetId);
  } catch (err) {
    console.error("Error selecting preset:", err);
    DOM.actionStatus.textContent = "Error loading preset";
  }
}

function updateSpectralBars(tel) {
  if (!tel) return;
  DOM.valDelta.textContent = `${tel.delta_pct}%`;
  DOM.barDelta.style.width = `${Math.min(100, tel.delta_pct)}%`;

  DOM.valTheta.textContent = `${tel.theta_pct}%`;
  DOM.barTheta.style.width = `${Math.min(100, tel.theta_pct)}%`;

  DOM.valAlpha.textContent = `${tel.alpha_pct}%`;
  DOM.barAlpha.style.width = `${Math.min(100, tel.alpha_pct)}%`;

  DOM.valBeta.textContent = `${tel.beta_pct}%`;
  DOM.barBeta.style.width = `${Math.min(100, tel.beta_pct)}%`;

  DOM.valGamma.textContent = `${tel.gamma_pct}%`;
  DOM.barGamma.style.width = `${Math.min(100, tel.gamma_pct)}%`;
}

// ==============================================================================
// Prediction & Staging Execution
// ==============================================================================

async function executePrediction(presetId = null) {
  DOM.actionStatus.textContent = "Evaluating multi-model consensus...";

  const payload = {
    preset_id: presetId || STATE.activePresetId,
    patient_id: STATE.activePatientId,
    consecutive_windows: STATE.consecutiveWindows,
    custom_sliders: {
      theta_beta_ratio: parseFloat(DOM.sliderTB.value),
      global_synchrony: parseFloat(DOM.sliderSync.value),
      hemispheric_asymmetry: parseFloat(DOM.sliderAsym.value),
      amplitude_scale: parseFloat(DOM.sliderAmp.value),
    },
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    STATE.lastPrediction = result;
    renderPredictionResults(result);
    DOM.actionStatus.textContent = `Completed • Scored by ${result.primary_model_used}`;
  } catch (err) {
    console.error("Prediction failed:", err);
    DOM.actionStatus.textContent = "Prediction error. Check server logs.";
  }
}

function renderPredictionResults(res) {
  const prob = res.probability;
  const pct = (prob * 100).toFixed(1);

  // 1. Stage Badge & Panel Theme
  DOM.stageState.textContent = res.predicted_stage;
  DOM.stageName.textContent = res.stage_label;
  DOM.modelUsedBadge.textContent = res.primary_model_used;

  DOM.stagePanel.className = `panel stage-panel glass-panel state-${res.color_theme}`;
  DOM.stageBadge.className = `stage-badge-large ${res.color_theme}`;

  // 2. Risk Level Pill & Lead Time
  DOM.riskLevelPill.textContent = `${res.risk_level} RISK`;
  DOM.riskLevelPill.className = `risk-pill pill-${res.color_theme}`;
  DOM.leadTimeVal.textContent = res.estimated_lead_time;

  // 3. Radial Risk Gauge Animation
  // Circumference: 2 * Math.PI * 50 = 314.16
  const circumference = 314.16;
  const offset = circumference * (1 - prob);
  DOM.gaugeCircle.style.strokeDashoffset = offset;
  DOM.gaugeVal.textContent = `${pct}%`;

  if (res.color_theme === "danger") {
    DOM.gaugeCircle.style.stroke = "var(--color-danger)";
  } else if (res.color_theme === "warning") {
    DOM.gaugeCircle.style.stroke = "var(--color-amber)";
  } else {
    DOM.gaugeCircle.style.stroke = "var(--color-emerald)";
  }

  // 4. Persistence Filter Card
  const pEval = res.persistence_evaluation;
  DOM.alarmStateBadge.textContent = pEval.alarm_active ? "ALARM ACTIVE" : "FILTERED / QUIET";
  DOM.alarmStateBadge.className = pEval.alarm_active ? "badge badge-danger" : "badge badge-success";
  DOM.persistenceDesc.textContent = pEval.filter_description;
  DOM.persistenceIcon.textContent = pEval.alarm_active ? "🚨" : "🛡️";

  if (prob >= 0.50) {
    DOM.step1.className = "step-pill step-active";
    DOM.step2.className = pEval.consecutive_windows_detected >= 2 ? "step-pill step-active" : "step-pill";
    DOM.step1.textContent = "Window t-1: Positive";
    DOM.step2.textContent = `Window t: Positive (${pct}%)`;
  } else {
    DOM.step1.className = "step-pill";
    DOM.step2.className = "step-pill";
    DOM.step1.textContent = "Window t-1: Interictal";
    DOM.step2.textContent = `Window t: Interictal (${pct}%)`;
  }

  // 5. Multi-Model Consensus Breakdown
  const consensus = res.multi_model_consensus;
  const pPers = consensus["Personalized Model (Calibrated 0.966 AUROC)"];
  const pRF = consensus["Random Forest Pipeline (453 Features)"];
  const pCNN = consensus["1D Convolutional Neural Network"];
  const pLSTM = consensus["Spatial-Temporal CNN-LSTM Candidate"];

  DOM.probPers.textContent = pPers !== null ? `${(pPers * 100).toFixed(1)}%` : "N/A";
  DOM.barPers.style.width = pPers !== null ? `${pPers * 100}%` : "0%";

  DOM.probRF.textContent = `${(pRF * 100).toFixed(1)}%`;
  DOM.barRF.style.width = `${pRF * 100}%`;

  DOM.probCNN.textContent = `${(pCNN * 100).toFixed(1)}%`;
  DOM.barCNN.style.width = `${pCNN * 100}%`;

  DOM.probLSTM.textContent = `${(pLSTM * 100).toFixed(1)}%`;
  DOM.barLSTM.style.width = `${pLSTM * 100}%`;

  // 6. Clinical Action Protocol
  DOM.protocolList.innerHTML = "";
  (res.clinical_protocol || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    DOM.protocolList.appendChild(li);
  });
}

// ==============================================================================
// 18-Channel Oscilloscope Canvas Engine
// ==============================================================================

function initCanvas() {
  const canvas = DOM.eegCanvas;
  const ctx = canvas.getContext("2d");

  // Adjust canvas pixel density
  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }
  resize();
  window.addEventListener("resize", resize);

  // Animation Loop
  function render() {
    drawOscilloscope(canvas, ctx);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

function drawOscilloscope(canvas, ctx) {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  // Clear background
  ctx.fillStyle = "#040711";
  ctx.fillRect(0, 0, width, height);

  const nChannels = STANDARD_CHANNELS.length;
  const channelHeight = height / nChannels;
  const userGain = STATE.gain * parseFloat(DOM.sliderAmp.value);

  // Draw each of the 18 channels
  STANDARD_CHANNELS.forEach((ch, idx) => {
    const baselineY = (idx + 0.5) * channelHeight;

    // Draw subtle channel divider
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, (idx + 1) * channelHeight);
    ctx.lineTo(width, (idx + 1) * channelHeight);
    ctx.stroke();

    // Channel Label (Montage layout colored by region)
    let labelColor = "rgba(148, 163, 184, 0.75)";
    if (ch.includes("FP1") || ch.includes("F7") || ch.includes("T7") || ch.includes("P7")) {
      labelColor = "#38bdf8"; // Left Temporal: cyan
    } else if (ch.includes("FP2") || ch.includes("F8") || ch.includes("T8") || ch.includes("P8")) {
      labelColor = "#c084fc"; // Right Temporal: purple
    } else if (ch.includes("FZ") || ch.includes("CZ") || ch.includes("PZ")) {
      labelColor = "#fbbf24"; // Midline: amber
    }

    ctx.fillStyle = labelColor;
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillText(ch, 10, baselineY - 4);

    // Waveform Trace
    const trace = STATE.waveformData[ch];
    if (trace && trace.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = labelColor;
      ctx.lineWidth = 1.25;

      const nPoints = trace.length;
      for (let i = 0; i < nPoints; i++) {
        const x = (i / (nPoints - 1)) * width;
        // Invert signal for electrophysiological display (- upwards)
        const y = baselineY - trace[i] * 0.45 * userGain;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
  });

  // Animated Scanning Sweep Line
  if (STATE.isSweeping) {
    STATE.sweepX = (STATE.sweepX + STATE.sweepSpeed) % width;
  }

  // Draw sweep bar with glow
  const sweepGradient = ctx.createLinearGradient(STATE.sweepX - 25, 0, STATE.sweepX, 0);
  sweepGradient.addColorStop(0, "rgba(14, 165, 233, 0)");
  sweepGradient.addColorStop(1, "rgba(14, 165, 233, 0.35)");

  ctx.fillStyle = sweepGradient;
  ctx.fillRect(STATE.sweepX - 25, 0, 25, height);

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(STATE.sweepX, 0);
  ctx.lineTo(STATE.sweepX, height);
  ctx.stroke();
}

// ==============================================================================
// Event Listeners
// ==============================================================================

function bindEvents() {
  // Gain Slider
  DOM.gainSlider.addEventListener("input", (e) => {
    STATE.gain = parseFloat(e.target.value);
    DOM.gainVal.textContent = `${STATE.gain.toFixed(1)}x`;
  });

  // Toggle Sweep
  DOM.btnToggleSweep.addEventListener("click", () => {
    STATE.isSweeping = !STATE.isSweeping;
    DOM.sweepIcon.textContent = STATE.isSweeping ? "⏸" : "▶";
    DOM.sweepText.textContent = STATE.isSweeping ? "Pause" : "Play";
  });

  // Slider Real-Time Numbers
  DOM.sliderTB.addEventListener("input", (e) => {
    DOM.valSliderTB.textContent = parseFloat(e.target.value).toFixed(2);
  });
  DOM.sliderSync.addEventListener("input", (e) => {
    DOM.valSliderSync.textContent = parseFloat(e.target.value).toFixed(2);
  });
  DOM.sliderAsym.addEventListener("input", (e) => {
    DOM.valSliderAsym.textContent = parseFloat(e.target.value).toFixed(2);
  });
  DOM.sliderAmp.addEventListener("input", (e) => {
    DOM.valSliderAmp.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
  });

  // Reset Sliders
  DOM.btnResetSliders.addEventListener("click", () => {
    if (STATE.customTelemetry) {
      DOM.sliderTB.value = STATE.customTelemetry.theta_beta_ratio || 1.89;
      DOM.valSliderTB.textContent = DOM.sliderTB.value;

      DOM.sliderSync.value = STATE.customTelemetry.global_synchrony || 0.45;
      DOM.valSliderSync.textContent = DOM.sliderSync.value;

      DOM.sliderAsym.value = STATE.customTelemetry.hemispheric_asymmetry || 1.05;
      DOM.valSliderAsym.textContent = DOM.sliderAsym.value;

      DOM.sliderAmp.value = 1.0;
      DOM.valSliderAmp.textContent = "1.00x";

      executePrediction();
    }
  });

  // File Upload Handlers
  DOM.btnUploadTrigger.addEventListener("click", () => DOM.fileInput.click());
  if (DOM.btnBrowseFile) {
    DOM.btnBrowseFile.addEventListener("click", (e) => {
      e.preventDefault();
      DOM.fileInput.click();
    });
  }

  DOM.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Drag and Drop
  ["dragenter", "dragover"].forEach((eventName) => {
    DOM.dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropZone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    DOM.dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropZone.classList.remove("drag-active");
    });
  });

  DOM.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  DOM.btnClearUploaded.addEventListener("click", () => {
    DOM.fileInput.value = "";
    DOM.uploadedBanner.style.display = "none";
    if (STATE.presets.length > 0) {
      selectPreset(STATE.presets[0].id);
    }
  });

  // Predict Button
  DOM.btnPredict.addEventListener("click", () => {
    executePrediction();
  });

  // Report Modal
  DOM.btnPrintReport.addEventListener("click", () => {
    openReportModal();
  });
  DOM.btnCloseModal.addEventListener("click", () => {
    DOM.reportModal.classList.remove("active");
  });
  DOM.btnCloseModal2.addEventListener("click", () => {
    DOM.reportModal.classList.remove("active");
  });
  DOM.btnPrintWindow.addEventListener("click", () => {
    window.print();
  });
}

async function handleFileUpload(file) {
  const name = file.name;
  const sizeMb = (file.size / (1024 * 1024)).toFixed(1);

  DOM.uploadProgressWrap.style.display = "block";
  DOM.uploadedBanner.style.display = "none";
  DOM.progressText.textContent = `Uploading and preprocessing '${name}' (${sizeMb} MB)...`;
  DOM.progressBarFill.style.width = "40%";
  DOM.progressPercent.textContent = "Ingesting";
  DOM.actionStatus.textContent = `Ingesting ${name}...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    DOM.progressBarFill.style.width = "75%";
    DOM.progressText.textContent = "Extracting 18-CH bipolar signals, computing features & staging...";
    DOM.progressPercent.textContent = "Predicting Stage";

    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || "Failed to process file");
    }

    DOM.progressBarFill.style.width = "100%";
    DOM.progressPercent.textContent = "Complete";

    setTimeout(() => {
      DOM.uploadProgressWrap.style.display = "none";
      DOM.uploadedBanner.style.display = "flex";
      DOM.uploadedFileName.textContent = data.filename;
      DOM.uploadedMeta.textContent = `• ${data.file_size_mb} MB • ${data.duration_minutes} mins (${data.total_windows_scored} Windows)`;
    }, 400);

    // Deselect preset cards
    document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("active"));
    STATE.activePresetId = null;
    STATE.activePatientId = data.patient_id;

    // Load authentic waveform from uploaded file into oscilloscope
    if (data.waveform_data) {
      STATE.waveformData = data.waveform_data;
      STATE.sweepX = 0;
    }

    // Update tags
    DOM.recordingSource.textContent = `${data.filename} • ${data.total_windows_scored} Windows`;
    DOM.canvasTimestamp.textContent = `File: ${data.filename} • Peak Window (${data.peak_window_time_min}m) • 18-CH Standard`;

    // Update spectral telemetry
    updateSpectralBars(data.telemetry_summary);

    // Update sliders
    DOM.sliderTB.value = data.telemetry_summary.theta_beta_ratio || 1.89;
    DOM.valSliderTB.textContent = DOM.sliderTB.value;
    DOM.sliderSync.value = data.telemetry_summary.global_synchrony || 0.45;
    DOM.valSliderSync.textContent = DOM.sliderSync.value;
    DOM.sliderAsym.value = data.telemetry_summary.hemispheric_asymmetry || 1.05;
    DOM.valSliderAsym.textContent = DOM.sliderAsym.value;

    // Render full predictions & staging
    STATE.lastPrediction = data;
    renderPredictionResults(data);

    DOM.actionStatus.textContent = `Stage prediction completed for '${data.filename}'!`;
  } catch (err) {
    console.error("Upload error:", err);
    DOM.uploadProgressWrap.style.display = "none";
    DOM.actionStatus.textContent = `Upload Error: ${err.message}`;
    alert(`File processing error: ${err.message}`);
  }
}

function openReportModal() {
  if (!STATE.lastPrediction) return;
  const p = STATE.lastPrediction;
  const date = new Date().toLocaleString();

  const reportText = `========================================================================
NEUROX • CLINICAL SEIZURE STAGING & PREDICTION REPORT
Generated: ${date}
Database: CHB-MIT Scalp EEG • International 10-20 Standard Montage (18 Channels)
========================================================================

1. PATIENT & TELEMETRY IDENTIFIERS:
------------------------------------------------------------------------
- Patient ID:          ${STATE.activePatientId.toUpperCase()}
- Active Recording:    ${DOM.recordingSource.textContent}
- Standard Channels:   18 Harmonized Longitudinal Bipolar Leads
- Primary Predictor:   ${p.primary_model_used}

2. CLINICAL STAGING EVALUATION:
------------------------------------------------------------------------
- Predicted Stage:     ${p.predicted_stage} (${p.stage_label})
- Preictal Probability: ${(p.probability * 100).toFixed(1)}%
- Clinical Risk Level: ${p.risk_level} ALERT
- Est. Warning Buffer: ${p.estimated_lead_time}

3. CLINICAL ALARM PERSISTENCE LOGIC (k = 2):
------------------------------------------------------------------------
- Consecutive Windows: ${p.persistence_evaluation.consecutive_windows_detected} / 2 required
- Clinical Alarm:      ${p.persistence_evaluation.alarm_active ? "ACTIVE (SUSTAINED PREICTAL TRANSITION)" : "SUPPRESSED / FILTERED"}
- Safety Logic:        ${p.persistence_evaluation.filter_description}

4. MULTI-MODEL CONSENSUS BREAKDOWN:
------------------------------------------------------------------------
- Personalized Model:  ${(p.multi_model_consensus["Personalized Model (Calibrated 0.966 AUROC)"] * 100).toFixed(1)}% (AUROC: 0.9664)
- Random Forest:       ${(p.multi_model_consensus["Random Forest Pipeline (453 Features)"] * 100).toFixed(1)}% (AUROC: 0.6135)
- 1D Convolutional NN: ${(p.multi_model_consensus["1D Convolutional Neural Network"] * 100).toFixed(1)}%
- Spatial-Temp CNN-LSTM: ${(p.multi_model_consensus["Spatial-Temporal CNN-LSTM Candidate"] * 100).toFixed(1)}% (Sensitivity: 82.65%)

5. NEUROPHYSIOLOGICAL METRICS:
------------------------------------------------------------------------
- Theta/Beta Ratio:    ${p.telemetry_summary.theta_beta_ratio}
- Cross Synchrony:     ${p.telemetry_summary.global_synchrony}
- Hemispheric Asym:    ${p.telemetry_summary.hemispheric_asymmetry}
- Dominant Focus:      ${p.telemetry_summary.dominant_focus}

6. RECOMMENDED CLINICAL PROTOCOL:
------------------------------------------------------------------------
${p.clinical_protocol.map((line) => `[!] ${line}`).join("\n")}
========================================================================`;

  DOM.modalReportContent.textContent = reportText;
  DOM.reportModal.classList.add("active");
}
