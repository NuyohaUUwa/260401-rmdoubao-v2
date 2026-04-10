from __future__ import annotations

import queue

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.job_manager import manager, sse_message

STATIC_VERSION = "20260407b"

app = FastAPI(title="Doubao Watermark Remover")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"title": "视频水印消除工具", "sv": STATIC_VERSION})


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/meta")
async def get_meta() -> dict:
    meta = manager.meta()
    return meta


@app.get("/api/system")
async def get_system() -> dict:
    meta = manager.meta()
    meta["queue"] = manager.queue_summary()
    return meta


@app.get("/api/queue")
async def get_queue() -> dict:
    return manager.queue_summary()


@app.get("/api/jobs")
async def list_jobs() -> dict:
    return manager.list_jobs()


@app.get("/api/history")
async def list_history() -> dict:
    return manager.list_history()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return job


@app.post("/api/jobs")
async def create_jobs(
    files: list[UploadFile] = File(...),
    preset: str = Form(...),
    use_gpu: str = Form("false"),
    keywords: str = Form("豆包,AI生成"),
    ffmpeg_path: str = Form(""),
    mask_mode: str = Form("char"),
    char_dilate: int = Form(1),
    char_blur: int = Form(3),
    track_gap: int = Form(8),
    min_instance_area: int = Form(12),
) -> dict:
    try:
        jobs = manager.create_jobs(
            files=files,
            preset=preset,
            use_gpu=parse_bool(use_gpu),
            keywords_raw=keywords,
            ffmpeg_path=ffmpeg_path,
            owner="public",
            mask_mode=mask_mode,
            char_dilate=char_dilate,
            char_blur=char_blur,
            track_gap=track_gap,
            min_instance_area=min_instance_area,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    subscriber = manager.subscribe_job(job_id)
    if subscriber is None:
        raise HTTPException(status_code=404, detail="任务不存在。")

    def event_stream():
        try:
            while True:
                try:
                    item = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield sse_message(item["event"], item["data"])
        finally:
            manager.unsubscribe_job(job_id, subscriber)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/queue/events")
async def stream_queue_events() -> StreamingResponse:
    subscriber = manager.subscribe_queue()

    def event_stream():
        try:
            while True:
                try:
                    item = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield sse_message(item["event"], item["data"])
        finally:
            manager.unsubscribe_queue(subscriber)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    info = manager.download_info(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="结果文件不存在或任务未完成。")
    path, media_type = info
    return FileResponse(path=path, media_type=media_type, filename=path.name)


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=10800, reload=False)


if __name__ == "__main__":
    main()
