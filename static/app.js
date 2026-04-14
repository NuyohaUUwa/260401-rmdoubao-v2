const state = {
  jobs: new Map(),
  historyJobs: new Map(),
  jobStreams: new Map(),
  queueStream: null,
  meta: null,
  advice: null,
};

const HISTORY_WINDOW_MS = 30 * 60 * 1000;
const CUSTOM_PRESET = "自定义";

function qs(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setSubmitMessage(message, isError = false) {
  const el = qs("submit-message");
  el.textContent = message;
  el.className = isError ? "submit-message error-text" : "submit-message";
}

function setAdviceMessage(message, isError = false) {
  const el = qs("advice-message");
  el.textContent = message;
  el.className = isError ? "submit-message error-text" : "submit-message";
}

function setAdviceFile(file, sourceLabel = "") {
  const fileInput = qs("advice-file");
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  updateAdviceFileStatus(sourceLabel || "已选择截图");
}

function updateAdviceFileStatus(prefix = "") {
  const status = qs("advice-file-status");
  const file = (qs("advice-file").files || [])[0];
  if (!file) {
    status.textContent = "选择或粘贴截图";
    return;
  }
  const info = `${file.name} (${Math.max(1, Math.round(file.size / 1024))} KB)`;
  status.textContent = prefix ? `${prefix}：${info}` : info;
}

function getHistoryBaseTime(job) {
  return job.finished_at || "";
}

function formatTimeLimit(job) {
  const baseTs = Date.parse(getHistoryBaseTime(job));
  if (Number.isNaN(baseTs)) {
    return "任务完成后保留 30 分钟，请及时下载保存。";
  }
  const remainMs = baseTs + HISTORY_WINDOW_MS - Date.now();
  if (remainMs <= 0) {
    return "已超出处理完成后的 30 分钟展示窗口。";
  }
  const remainMinutes = Math.max(1, Math.ceil(remainMs / 60000));
  return `处理完成后保留 30 分钟，请在 ${remainMinutes} 分钟内及时下载保存。`;
}

function isVisibleHistoryJob(job) {
  const baseTs = Date.parse(getHistoryBaseTime(job));
  return !Number.isNaN(baseTs) && Date.now() - baseTs <= HISTORY_WINDOW_MS;
}

function updateQueueSummary(summary) {
  qs("queue-running").textContent = summary.running;
  qs("queue-waiting").textContent = summary.queued;
  qs("queue-remaining").textContent = summary.remaining_capacity;
}

function statusBadge(job) {
  const cls = job.status === "failed" ? "badge failed" : job.status === "queued" ? "badge queued" : "badge";
  return `<span class="${cls}">${job.stage_label}</span>`;
}

function renderJobs() {
  const container = qs("jobs");
  const jobs = Array.from(state.jobs.values()).sort((a, b) => a.sequence_code.localeCompare(b.sequence_code));
  if (!jobs.length) {
    container.className = "jobs jobs-scroll empty";
    container.textContent = "暂无任务";
    return;
  }
  container.className = "jobs jobs-scroll";
  container.innerHTML = jobs.map((job) => {
    const queueText = job.queue_position ? `排队位置：${job.queue_position}，前方 ${job.ahead_in_queue} 个任务` : "当前不在排队";
    const progressText = `${job.progress.current || 0}/${job.progress.total || 0} (${job.progress.percent || 0}%)`;
    const errorText = job.error ? `<p class="meta-line error-text">${job.error}</p>` : "";
    const download = job.download_url
      ? `<a class="download-link" href="${job.download_url}">下载结果</a>`
      : "";
    return `
      <article class="job-card">
        <div class="job-head">
          <div>
            <h3 class="job-title">${job.filename}</h3>
            <p class="meta-line">序列码：${job.sequence_code} | 任务ID：${job.job_id}</p>
          </div>
          ${statusBadge(job)}
        </div>
        <p class="meta-line">阶段：${job.stage_label}</p>
        <p class="meta-line">消息：${job.message}</p>
        <p class="meta-line">${queueText}</p>
        <p class="meta-line">进度：${progressText}</p>
        <div class="progress-track">
          <div class="progress-bar" style="width:${job.progress.percent || 0}%"></div>
        </div>
        <div class="job-actions">
          ${download}
        </div>
        ${errorText}
      </article>
    `;
  }).join("");
}

function upsertJob(job) {
  if (job.status === "succeeded" || job.status === "failed") {
    state.jobs.delete(job.job_id);
    state.historyJobs.set(job.job_id, job);
    renderHistoryJobs();
  } else {
    state.jobs.set(job.job_id, job);
  }
  renderJobs();
}

function renderHistoryJobs() {
  const container = qs("history-jobs");
  const jobs = Array.from(state.historyJobs.values())
    .filter(isVisibleHistoryJob)
    .sort((a, b) => b.sequence_code.localeCompare(a.sequence_code));
  if (!jobs.length) {
    container.className = "jobs jobs-scroll empty";
    container.textContent = "暂无 30 分钟内的历史视频";
    return;
  }
  container.className = "jobs jobs-scroll";
  container.innerHTML = jobs.map((job) => {
    const progressText = `${job.progress.current || 0}/${job.progress.total || 0} (${job.progress.percent || 0}%)`;
    const errorText = job.error ? `<p class="meta-line error-text">${job.error}</p>` : "";
    const download = job.download_url
      ? `<a class="download-link" href="${job.download_url}">下载结果</a>`
      : "";
    return `
      <article class="job-card">
        <div class="job-head">
          <div>
            <h3 class="job-title">${job.filename}</h3>
            <p class="meta-line">序列码：${job.sequence_code} | 任务ID：${job.job_id}</p>
          </div>
          ${statusBadge(job)}
        </div>
        <p class="meta-line">阶段：${job.stage_label}</p>
        <p class="meta-line">消息：${job.message}</p>
        <p class="meta-line download-reminder">下载提醒：${formatTimeLimit(job)}</p>
        <p class="meta-line">进度：${progressText}</p>
        <div class="job-actions">${download}</div>
        ${errorText}
      </article>
    `;
  }).join("");
}

function subscribeQueue() {
  if (state.queueStream) {
    state.queueStream.close();
  }
  const stream = new EventSource("/api/queue/events");
  stream.addEventListener("queue.updated", (event) => {
    updateQueueSummary(JSON.parse(event.data));
  });
  state.queueStream = stream;
}

function subscribeJob(jobId) {
  if (state.jobStreams.has(jobId)) {
    return;
  }
  const stream = new EventSource(`/api/jobs/${jobId}/events`);
  const events = [
    "job.snapshot",
    "job.queued",
    "job.started",
    "job.stage_changed",
    "job.progress",
    "job.succeeded",
    "job.failed",
  ];
  for (const eventName of events) {
    stream.addEventListener(eventName, (event) => {
      const job = JSON.parse(event.data);
      upsertJob(job);
      if (job.status === "succeeded" || job.status === "failed") {
        stream.close();
        state.jobStreams.delete(jobId);
      }
    });
  }
  state.jobStreams.set(jobId, stream);
}

const PRESET_LABELS = {
  "标准（推荐）": "标准（推荐） - 均衡",
  "更干净": "更干净 - 清理更彻底",
  "更清晰": "更清晰 - 更保细节",
  "更快速": "更快速 - 提速优先",
  [CUSTOM_PRESET]: "自定义 - 手动设置参数",
};

function maskModeForPreset(preset) {
  return preset === "更快速" ? "rect" : state.meta.default_custom_mask_mode;
}

function setCustomFieldsEnabled(enabled) {
  ["custom-pad", "custom-radius", "custom-crf"].forEach((id) => {
    qs(id).disabled = !enabled;
  });
}

function applyPresetDefaults(preset) {
  const defaults = preset === CUSTOM_PRESET ? state.meta.custom_defaults : state.meta.presets[preset];
  qs("custom-pad").value = defaults.pad;
  qs("custom-radius").value = defaults.radius;
  qs("custom-crf").value = defaults.crf;
  qs("mask-mode").value = maskModeForPreset(preset);
  setCustomFieldsEnabled(preset === CUSTOM_PRESET);
}

function parseNonNegativeInteger(value, label) {
  if (value === "") {
    return null;
  }
  if (!/^\d+$/.test(value)) {
    throw new Error(`${label}必须是大于等于 0 的整数。`);
  }
  return Number.parseInt(value, 10);
}

function parseBoundedInteger(value, label, min, max) {
  if (!/^-?\d+$/.test(value)) {
    throw new Error(`${label}必须是整数。`);
  }
  const parsed = Number.parseInt(value, 10);
  if (parsed < min || parsed > max) {
    throw new Error(`${label}必须在 ${min} 到 ${max} 之间。`);
  }
  return parsed;
}

function validateAdvancedOptions() {
  const preset = qs("preset").value;
  const result = {
    preset,
    maskMode: qs("mask-mode").value,
    customPad: null,
    customRadius: null,
    customCrf: null,
    frameStart: null,
    frameEnd: null,
  };
  if (preset === CUSTOM_PRESET) {
    result.customPad = parseBoundedInteger(qs("custom-pad").value.trim(), "pad", 0, 64);
    result.customRadius = parseBoundedInteger(qs("custom-radius").value.trim(), "radius", 1, 32);
    result.customCrf = parseBoundedInteger(qs("custom-crf").value.trim(), "crf", 0, 51);
  }
  const frameStartRaw = qs("frame-start").value.trim();
  const frameEndRaw = qs("frame-end").value.trim();
  const frameStart = parseNonNegativeInteger(frameStartRaw, "起始帧");
  const frameEnd = parseNonNegativeInteger(frameEndRaw, "结束帧");
  if ((frameStart === null) !== (frameEnd === null)) {
    throw new Error("起始帧和结束帧需要同时填写，留空则表示全视频。");
  }
  if (frameStart !== null && frameEnd !== null && frameStart > frameEnd) {
    throw new Error("起始帧不能大于结束帧。");
  }
  result.frameStart = frameStart;
  result.frameEnd = frameEnd;
  return result;
}

function renderAdviceResult() {
  const container = qs("advice-result");
  if (!state.advice) {
    container.className = "advice-result empty";
    container.textContent = "暂无分析结果";
    return;
  }
  const { message, recommended, preview_url: previewUrl, recognized_texts: texts, boxes } = state.advice;
  const textSummary = texts.length ? texts.map((text) => escapeHtml(text)).join(" / ") : "未识别到明确的匹配文字";
  const boxSummary = boxes.length
    ? boxes.map((box, index) => `#${index + 1} (${box.x0}, ${box.y0}) - (${box.x1}, ${box.y1})`).map((text) => escapeHtml(text)).join("<br>")
    : "未识别到明确框选区域";
  container.className = "advice-result";
  container.innerHTML = `
    <div class="advice-card">
      <p class="meta-line">${escapeHtml(message)}</p>
      <div class="advice-grid">
        <div><span>pad</span><strong>${escapeHtml(recommended.pad)}</strong></div>
        <div><span>radius</span><strong>${escapeHtml(recommended.radius)}</strong></div>
        <div><span>crf</span><strong>${escapeHtml(recommended.crf)}</strong></div>
        <div><span>掩膜模式</span><strong>${escapeHtml(recommended.mask_mode)}</strong></div>
        <div><span>预设倾向</span><strong>${escapeHtml(recommended.preset_hint)}</strong></div>
      </div>
      <p class="meta-line">识别文字：${textSummary}</p>
      <p class="meta-line">框位置：${boxSummary}</p>
      <div class="advice-actions">
        <button id="apply-advice-btn" type="button">一键填入主表单</button>
      </div>
    </div>
    <div class="advice-preview">
      <img src="${escapeHtml(previewUrl)}" alt="截图分析标注预览">
    </div>
  `;
  qs("apply-advice-btn").addEventListener("click", applyAdviceToForm);
}

function applyAdviceToForm() {
  if (!state.advice) {
    return;
  }
  const { recommended } = state.advice;
  qs("preset").value = CUSTOM_PRESET;
  applyPresetDefaults(CUSTOM_PRESET);
  qs("custom-pad").value = recommended.pad;
  qs("custom-radius").value = recommended.radius;
  qs("custom-crf").value = recommended.crf;
  qs("mask-mode").value = recommended.mask_mode;
  setSubmitMessage("已将建议参数填入主表单。");
}

async function loadInitialState() {
  const [metaResp, jobsResp] = await Promise.all([
    fetch("/api/meta"),
    fetch("/api/jobs"),
  ]);
  state.meta = await metaResp.json();
  const jobsData = await jobsResp.json();

  const presetSelect = qs("preset");
  Object.keys(state.meta.presets).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = PRESET_LABELS[name] || name;
    if (name === state.meta.default_preset) {
      option.selected = true;
    }
    presetSelect.appendChild(option);
  });
  const customOption = document.createElement("option");
  customOption.value = state.meta.custom_preset;
  customOption.textContent = PRESET_LABELS[state.meta.custom_preset] || state.meta.custom_preset;
  presetSelect.appendChild(customOption);

  const maskModeSelect = qs("mask-mode");
  state.meta.mask_modes.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    if (name === state.meta.default_mask_mode) {
      option.selected = true;
    }
    maskModeSelect.appendChild(option);
  });

  qs("keywords").value = state.meta.default_keywords;
  qs("use-gpu").checked = state.meta.gpu_available;
  qs("gpu-hint").textContent = state.meta.gpu_available ? "已检测到 CUDA" : "未检测到 CUDA，将使用 CPU";
  applyPresetDefaults(state.meta.default_preset);

  updateQueueSummary(jobsData.queue);
  for (const job of jobsData.jobs) {
    upsertJob(job);
    if (job.status === "queued" || job.status === "running") {
      subscribeJob(job.job_id);
    }
  }
  await loadHistory();
}

async function loadHistory() {
  const historyResp = await fetch("/api/history");
  const historyData = await historyResp.json();
  state.historyJobs.clear();
  for (const job of historyData.jobs || []) {
    state.historyJobs.set(job.job_id, job);
  }
  renderHistoryJobs();
}

async function submitJobs(event) {
  event.preventDefault();
  const filesInput = qs("files");
  const files = Array.from(filesInput.files || []);
  if (!files.length) {
    setSubmitMessage("请先选择视频文件。", true);
    return;
  }
  if (files.length > 3) {
    setSubmitMessage("单次最多上传 3 个视频。", true);
    return;
  }

  let options;
  try {
    options = validateAdvancedOptions();
  } catch (error) {
    setSubmitMessage(error.message || "高级设置参数无效。", true);
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("preset", options.preset);
  formData.append("use_gpu", qs("use-gpu").checked ? "true" : "false");
  formData.append("keywords", qs("keywords").value);
  formData.append("ffmpeg_path", qs("ffmpeg-path").value);
  formData.append("mask_mode", options.maskMode);
  if (options.preset === CUSTOM_PRESET) {
    formData.append("custom_pad", String(options.customPad));
    formData.append("custom_radius", String(options.customRadius));
    formData.append("custom_crf", String(options.customCrf));
  }
  if (options.frameStart !== null && options.frameEnd !== null) {
    formData.append("frame_start", String(options.frameStart));
    formData.append("frame_end", String(options.frameEnd));
  }

  const submitBtn = qs("submit-btn");
  submitBtn.disabled = true;
  setSubmitMessage("正在提交任务...");

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "提交失败");
    }
    for (const job of data.jobs) {
      upsertJob(job);
      subscribeJob(job.job_id);
    }
    setSubmitMessage(`已提交 ${data.jobs.length} 个任务。`);
    filesInput.value = "";
  } catch (error) {
    setSubmitMessage(error.message || "提交失败", true);
  } finally {
    submitBtn.disabled = false;
  }
}

async function submitAdvice(event) {
  event?.preventDefault();
  const fileInput = qs("advice-file");
  const file = (fileInput.files || [])[0];
  if (!file) {
    setAdviceMessage("请先选择一张截图。", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  const submitBtn = qs("advice-submit-btn");
  submitBtn.disabled = true;
  setAdviceMessage("正在分析截图...");

  try {
    const response = await fetch("/api/screenshot-advice", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "截图分析失败");
    }
    state.advice = data;
    renderAdviceResult();
    setAdviceMessage("截图分析完成。");
  } catch (error) {
    state.advice = null;
    renderAdviceResult();
    setAdviceMessage(error.message || "截图分析失败", true);
  } finally {
    submitBtn.disabled = false;
  }
}

function handleAdvicePaste(event) {
  const items = Array.from(event.clipboardData?.items || []);
  const imageItem = items.find((item) => item.kind === "file" && item.type.startsWith("image/"));
  if (!imageItem) {
    return false;
  }
  const file = imageItem.getAsFile();
  if (!file) {
    return false;
  }
  const ext = (file.type.split("/")[1] || "png").replace("jpeg", "jpg");
  const normalizedFile = new File([file], `pasted-screenshot.${ext}`, { type: file.type || "image/png" });
  setAdviceFile(normalizedFile, "已粘贴");
  setAdviceMessage("截图已粘贴。");
  return true;
}

function bindAdvicePaste() {
  const fileInput = qs("advice-file");
  const pasteZone = qs("advice-paste-zone");

  fileInput.addEventListener("change", () => {
    updateAdviceFileStatus("已选择截图");
  });

  pasteZone.addEventListener("click", () => {
    pasteZone.focus();
  });

  pasteZone.addEventListener("paste", (event) => {
    if (handleAdvicePaste(event)) {
      event.preventDefault();
    }
  });

  document.addEventListener("paste", (event) => {
    const active = document.activeElement;
    const isTypingTarget = active && (
      active.tagName === "INPUT"
      || active.tagName === "TEXTAREA"
      || active.isContentEditable
      || active.tagName === "SELECT"
    );
    if (isTypingTarget && active !== pasteZone) {
      return;
    }
    if (handleAdvicePaste(event)) {
      event.preventDefault();
    }
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  qs("job-form").addEventListener("submit", submitJobs);
  qs("advice-submit-btn").addEventListener("click", submitAdvice);
  qs("preset").addEventListener("change", (event) => {
    applyPresetDefaults(event.target.value);
  });
  bindAdvicePaste();
  try {
    subscribeQueue();
    await loadInitialState();
    renderAdviceResult();
    updateAdviceFileStatus();
    window.setInterval(renderHistoryJobs, 30000);
  } catch (error) {
    setSubmitMessage("初始化失败，请刷新页面重试。", true);
  }
});
