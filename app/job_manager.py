from __future__ import annotations

import json
import mimetypes
import queue
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

from core.processor import infer_torch_cuda, mode_auto, repo_root, resolve_ffmpeg

PRESETS: dict[str, dict[str, int]] = {
    "标准（推荐）": {"pad": 18, "radius": 5, "crf": 18},
    "更干净": {"pad": 20, "radius": 5, "crf": 17},
    "更清晰": {"pad": 18, "radius": 4, "crf": 16},
    "更快速": {"pad": 18, "radius": 5, "crf": 21},
}

MAX_BATCH_UPLOADS = 3
MAX_QUEUE_SIZE = 5
DEFAULT_KEYWORDS = "豆包,AI生成"
STAGE_LABELS = {
    "uploaded": "已上传",
    "queued": "排队中",
    "initializing_ocr": "初始化 OCR",
    "scanning": "扫描水印位置",
    "repairing": "修复视频帧",
    "muxing": "封装输出视频",
    "validating": "校验输出",
    "completed": "完成",
    "failed": "失败",
}


def safe_output_stem(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*\x00-\x1F]+', "_", name).strip(" .")
    return cleaned or "video"


@dataclass
class JobProgress:
    current: int = 0
    total: int = 0
    percent: float = 0.0


@dataclass
class JobRecord:
    job_id: str
    sequence_code: str
    filename: str
    input_path: str
    output_path: str
    preset: str
    keywords: list[str]
    use_gpu: bool
    ffmpeg_path: str | None
    owner: str = "user"
    mask_mode: str = "char"
    char_dilate: int = 1
    char_blur: int = 3
    track_gap: int = 8
    min_instance_area: int = 12
    status: str = "queued"
    stage: str = "uploaded"
    stage_label: str = STAGE_LABELS["uploaded"]
    message: str = "已上传"
    error: str | None = None
    progress: JobProgress = field(default_factory=JobProgress)


class JobManager:
    def __init__(self) -> None:
        self.root = repo_root()
        self.upload_root = self.root / "uploads"
        self.output_root = self.root / "output"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._pending: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._job_subscribers: dict[str, list[queue.Queue[dict[str, Any]]]] = {}
        self._queue_subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._sequence = count(1)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def meta(self) -> dict[str, Any]:
        return {
            "presets": PRESETS,
            "default_preset": "标准（推荐）",
            "default_keywords": DEFAULT_KEYWORDS,
            "gpu_available": infer_torch_cuda(),
            "max_batch_uploads": MAX_BATCH_UPLOADS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "stage_labels": STAGE_LABELS,
            "mask_modes": ["char", "rect"],
            "default_mask_mode": "char",
        }

    @staticmethod
    def _can_access(job: JobRecord, username: str, role: str) -> bool:
        return role == "admin" or job.owner == username

    def _visible_job_ids(self, username: str, role: str) -> list[str]:
        return [
            job_id
            for job_id, job in self._jobs.items()
            if self._can_access(job, username, role)
        ]

    def create_jobs(
        self,
        files: list[Any],
        preset: str,
        use_gpu: bool,
        keywords_raw: str,
        ffmpeg_path: str | None,
        owner: str,
        mask_mode: str = "char",
        char_dilate: int = 1,
        char_blur: int = 3,
        track_gap: int = 8,
        min_instance_area: int = 12,
    ) -> list[dict[str, Any]]:
        if not files:
            raise ValueError("请至少上传 1 个视频文件。")
        if len(files) > MAX_BATCH_UPLOADS:
            raise ValueError(f"单次最多上传 {MAX_BATCH_UPLOADS} 个视频。")
        if preset not in PRESETS:
            raise ValueError("无效的预设。")
        if mask_mode not in {"char", "rect"}:
            raise ValueError("无效的掩膜模式，仅支持 char 或 rect。")
        if preset == "更快速":
            # “更快速”固定走修改前的矩形方案。
            mask_mode = "rect"
            char_dilate = 0
            char_blur = 1
            track_gap = 4
            min_instance_area = 16

        keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        if not keywords:
            keywords = [k.strip() for k in DEFAULT_KEYWORDS.split(",") if k.strip()]
        char_dilate = max(0, min(8, int(char_dilate)))
        char_blur = max(1, min(15, int(char_blur)))
        track_gap = max(2, min(24, int(track_gap)))
        min_instance_area = max(4, min(128, int(min_instance_area)))

        with self._lock:
            inflight = sum(1 for job in self._jobs.values() if job.status in {"queued", "running"})
            if inflight + len(files) > MAX_QUEUE_SIZE:
                raise ValueError(f"当前队列最多容纳 {MAX_QUEUE_SIZE} 个任务，请稍后再试。")

            created: list[JobRecord] = []
            for upload in files:
                job_id = uuid.uuid4().hex
                sequence_code = f"{next(self._sequence):03d}"
                original_name = Path(getattr(upload, "filename", "video.mp4") or "video.mp4").name
                upload_dir = self.upload_root / job_id
                output_dir = self.output_root / job_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)
                input_path = upload_dir / original_name
                output_name = f"去水印_{safe_output_stem(Path(original_name).stem)}_{sequence_code}.mp4"
                output_path = output_dir / output_name

                with input_path.open("wb") as f:
                    shutil.copyfileobj(upload.file, f)

                job = JobRecord(
                    job_id=job_id,
                    sequence_code=sequence_code,
                    filename=original_name,
                    input_path=str(input_path),
                    output_path=str(output_path),
                    preset=preset,
                    keywords=keywords,
                    use_gpu=use_gpu,
                    ffmpeg_path=ffmpeg_path.strip() if ffmpeg_path and ffmpeg_path.strip() else None,
                    owner=owner,
                    mask_mode=mask_mode,
                    char_dilate=char_dilate,
                    char_blur=char_blur,
                    track_gap=track_gap,
                    min_instance_area=min_instance_area,
                )
                self._jobs[job_id] = job
                self._job_subscribers[job_id] = []
                created.append(job)

            for job in created:
                self._pending.put(job.job_id)
                self._set_job_stage(job.job_id, "queued", "排队中", status="queued", event="job.queued")

            self._broadcast_queue()
            return [self._serialize_job(job.job_id) for job in created]

    def get_job(self, job_id: str, username: str, role: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not self._can_access(job, username, role):
                return None
            return self._serialize_job(job_id)

    def list_jobs(self, username: str, role: str) -> dict[str, Any]:
        with self._lock:
            jobs = [
                self._serialize_job(job_id)
                for job_id in self._visible_job_ids(username, role)
                if self._jobs[job_id].status in {"queued", "running"}
            ]
            jobs.sort(key=lambda item: item["sequence_code"])
            return {
                "jobs": jobs,
                "queue": self._queue_summary_locked(),
            }

    def list_history(self, username: str, role: str) -> dict[str, Any]:
        with self._lock:
            jobs = [
                self._serialize_job(job_id)
                for job_id in self._visible_job_ids(username, role)
                if self._jobs[job_id].status in {"succeeded", "failed"}
            ]
            jobs.sort(key=lambda item: item["sequence_code"], reverse=True)
            return {"jobs": jobs}

    def queue_summary(self) -> dict[str, int]:
        with self._lock:
            return self._queue_summary_locked()

    def subscribe_job(self, job_id: str, username: str, role: str) -> queue.Queue[dict[str, Any]] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not self._can_access(job, username, role):
                return None
            q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=50)
            self._job_subscribers[job_id].append(q)
            q.put({"event": "job.snapshot", "data": self._serialize_job(job_id)})
            return q

    def unsubscribe_job(self, job_id: str, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subscribers = self._job_subscribers.get(job_id)
            if subscribers and q in subscribers:
                subscribers.remove(q)

    def subscribe_queue(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=50)
        with self._lock:
            self._queue_subscribers.append(q)
            q.put({"event": "queue.updated", "data": self._queue_summary_locked()})
        return q

    def unsubscribe_queue(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._queue_subscribers:
                self._queue_subscribers.remove(q)

    def download_info(self, job_id: str, username: str, role: str) -> tuple[Path, str] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "succeeded" or not self._can_access(job, username, role):
                return None
            path = Path(job.output_path)
            if not path.is_file():
                return None
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return path, mime

    def _worker_loop(self) -> None:
        while True:
            job_id = self._pending.get()
            try:
                self._run_job(job_id)
            finally:
                self._pending.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
        try:
            ffmpeg = resolve_ffmpeg(job.ffmpeg_path)
            self._set_job_stage(job_id, "initializing_ocr", "初始化 OCR", status="running", event="job.started")

            preset = PRESETS[job.preset]

            def on_stage(stage: str, message: str) -> None:
                self._set_job_stage(job_id, stage, message, status="running", event="job.stage_changed")

            def on_scan_progress(current: int, total: int) -> None:
                percent = min(100.0, round((current / max(total, 1)) * 40.0, 2))
                with self._lock:
                    current_job = self._jobs[job_id]
                    current_job.progress.current = current
                    current_job.progress.total = total
                    current_job.progress.percent = percent
                    current_job.message = f"扫描中 {current}/{total}"
                    payload = self._serialize_job(job_id)
                self._emit_job(job_id, "job.progress", payload)

            def on_progress(current: int, total: int) -> None:
                percent = min(100.0, round(40.0 + (current / max(total, 1)) * 60.0, 2))
                with self._lock:
                    current_job = self._jobs[job_id]
                    current_job.progress.current = current
                    current_job.progress.total = total
                    current_job.progress.percent = percent
                    current_job.message = f"处理中 {current}/{total}"
                    payload = self._serialize_job(job_id)
                self._emit_job(job_id, "job.progress", payload)

            mode_auto(
                input_path=Path(job.input_path),
                output_path=Path(job.output_path),
                ffmpeg=ffmpeg,
                keywords=job.keywords,
                pad=preset["pad"],
                inpaint_radius=preset["radius"],
                crf=preset["crf"],
                ocr_interval=1,
                carry_bbox=True,
                progress=on_progress,
                use_gpu=job.use_gpu,
                roi_fallback=True,
                stage_callback=on_stage,
                scan_progress=on_scan_progress,
                mask_mode=job.mask_mode,
                char_dilate=job.char_dilate,
                char_blur=job.char_blur,
                track_gap=job.track_gap,
                min_instance_area=job.min_instance_area,
            )
            self._set_job_stage(job_id, "completed", "完成", status="succeeded", event="job.succeeded")
        except Exception as exc:
            with self._lock:
                current_job = self._jobs[job_id]
                current_job.status = "failed"
                current_job.stage = "failed"
                current_job.stage_label = STAGE_LABELS["failed"]
                current_job.message = "处理失败"
                current_job.error = str(exc)
                payload = self._serialize_job(job_id)
            self._emit_job(job_id, "job.failed", payload)
        finally:
            self._broadcast_queue()

    def _set_job_stage(
        self,
        job_id: str,
        stage: str,
        message: str,
        *,
        status: str,
        event: str,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.stage = stage
            job.stage_label = STAGE_LABELS.get(stage, message)
            job.message = message
            if status == "succeeded":
                job.progress.percent = 100.0
                if job.progress.total > 0:
                    job.progress.current = job.progress.total
            payload = self._serialize_job(job_id)
        self._emit_job(job_id, event, payload)
        self._broadcast_queue()

    def _serialize_job(self, job_id: str) -> dict[str, Any]:
        job = self._jobs[job_id]
        queue_position = self._queue_position_locked(job_id)
        download_url = f"/api/jobs/{job.job_id}/download" if job.status == "succeeded" else None
        data = asdict(job)
        data["progress"] = asdict(job.progress)
        data["queue_position"] = queue_position
        data["ahead_in_queue"] = max(0, queue_position - 1) if queue_position else 0
        data["download_url"] = download_url
        return data

    def _queue_summary_locked(self) -> dict[str, int]:
        running = sum(1 for job in self._jobs.values() if job.status == "running")
        queued = sum(1 for job in self._jobs.values() if job.status == "queued")
        inflight = running + queued
        return {
            "running": running,
            "queued": queued,
            "capacity": MAX_QUEUE_SIZE,
            "remaining_capacity": max(0, MAX_QUEUE_SIZE - inflight),
        }

    def _queue_position_locked(self, job_id: str) -> int | None:
        job = self._jobs[job_id]
        if job.status != "queued":
            return None
        queued_jobs = sorted(
            (item for item in self._jobs.values() if item.status == "queued"),
            key=lambda item: item.sequence_code,
        )
        for index, item in enumerate(queued_jobs, start=1):
            if item.job_id == job_id:
                return index
        return None

    def _emit_job(self, job_id: str, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._job_subscribers.get(job_id, []))
        for subscriber in subscribers:
            try:
                subscriber.put_nowait({"event": event, "data": data})
            except queue.Full:
                pass

    def _broadcast_queue(self) -> None:
        with self._lock:
            summary = self._queue_summary_locked()
            subscribers = list(self._queue_subscribers)
            queued_payloads = [
                (job.job_id, self._serialize_job(job.job_id))
                for job in self._jobs.values()
                if job.status == "queued"
            ]
        for subscriber in subscribers:
            try:
                subscriber.put_nowait({"event": "queue.updated", "data": summary})
            except queue.Full:
                pass
        for job_id, payload in queued_payloads:
            self._emit_job(job_id, "job.snapshot", payload)


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


manager = JobManager()
