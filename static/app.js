const state = {
  jobs: new Map(),
  jobStreams: new Map(),
  queueStream: null,
  meta: null,
};

function qs(id) {
  return document.getElementById(id);
}

function setSubmitMessage(message, isError = false) {
  const el = qs("submit-message");
  el.textContent = message;
  el.className = isError ? "submit-message error-text" : "submit-message";
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
    container.className = "jobs empty";
    container.textContent = "暂无任务";
    return;
  }
  container.className = "jobs";
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
  state.jobs.set(job.job_id, job);
  renderJobs();
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
    option.textContent = name;
    if (name === state.meta.default_preset) {
      option.selected = true;
    }
    presetSelect.appendChild(option);
  });

  qs("keywords").value = state.meta.default_keywords;
  qs("use-gpu").checked = state.meta.gpu_available;
  qs("gpu-hint").textContent = state.meta.gpu_available ? "已检测到 CUDA" : "未检测到 CUDA，将使用 CPU";

  updateQueueSummary(jobsData.queue);
  for (const job of jobsData.jobs) {
    upsertJob(job);
    if (job.status === "queued" || job.status === "running") {
      subscribeJob(job.job_id);
    }
  }
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

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("preset", qs("preset").value);
  formData.append("use_gpu", qs("use-gpu").checked ? "true" : "false");
  formData.append("keywords", qs("keywords").value);
  formData.append("ffmpeg_path", qs("ffmpeg-path").value);

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

document.addEventListener("DOMContentLoaded", async () => {
  qs("job-form").addEventListener("submit", submitJobs);
  subscribeQueue();
  try {
    await loadInitialState();
  } catch (error) {
    setSubmitMessage("初始化失败，请刷新页面重试。", true);
  }
});
