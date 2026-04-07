from __future__ import annotations

import base64
import binascii
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.job_manager import manager, sse_message

ROOT = Path(__file__).resolve().parent.parent
USERS_FILE = ROOT / "users.json"
USERS_LOCK = threading.RLock()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def decode_base64_password(encoded: str) -> str:
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise RuntimeError("base64 密码格式无效。") from exc


def encode_base64_password(plain: str) -> str:
    return base64.b64encode(plain.encode("utf-8")).decode("utf-8")


def load_users_from_file(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    users: dict[str, dict[str, str]] = {}
    if not isinstance(payload, dict):
        return users
    for username, item in payload.items():
        if not isinstance(username, str) or not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip().lower()
        password_b64 = str(item.get("password_b64", "")).strip()
        if role not in {"user", "admin"} or not password_b64:
            continue
        try:
            password = decode_base64_password(password_b64)
        except RuntimeError:
            continue
        users[username.strip()] = {"password": password, "role": role}
    return users


def save_users_to_file(path: Path, users: dict[str, dict[str, str]]) -> None:
    payload: dict[str, dict[str, str]] = {}
    for username, item in users.items():
        role = str(item.get("role", "user"))
        if role not in {"user", "admin"}:
            continue
        payload[username] = {
            "role": role,
            "password_b64": encode_base64_password(str(item.get("password", ""))),
        }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


load_env_file(ROOT / ".env")

STATIC_VERSION = "20260407b"

app = FastAPI(title="Doubao Watermark Remover")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("DOUBAO_SESSION_SECRET", "doubao-dev-secret"),
    same_site="lax",
    https_only=False,
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

USERS: dict[str, dict[str, str]] = load_users_from_file(USERS_FILE)


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def require_user(request: Request) -> tuple[str, str]:
    username = str(request.session.get("username") or "").strip()
    role = str(request.session.get("role") or "").strip()
    if not username or role not in {"user", "admin"}:
        raise HTTPException(status_code=401, detail="请先登录。")
    return username, role


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    if not request.session.get("username"):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "index.html", {"title": "视频水印消除工具", "sv": STATIC_VERSION})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if request.session.get("username"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"title": "登录 - 视频水印消除工具", "sv": STATIC_VERSION})


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/meta")
async def get_meta(request: Request) -> dict:
    username, role = require_user(request)
    meta = manager.meta()
    meta["user"] = {"username": username, "role": role}
    return meta


@app.get("/api/system")
async def get_system(request: Request) -> dict:
    username, role = require_user(request)
    meta = manager.meta()
    meta["queue"] = manager.queue_summary()
    meta["user"] = {"username": username, "role": role}
    return meta


@app.get("/api/queue")
async def get_queue(request: Request) -> dict:
    require_user(request)
    return manager.queue_summary()


@app.get("/api/jobs")
async def list_jobs(request: Request) -> dict:
    username, role = require_user(request)
    return manager.list_jobs(username, role)


@app.get("/api/history")
async def list_history(request: Request) -> dict:
    username, role = require_user(request)
    return manager.list_history(username, role)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict:
    username, role = require_user(request)
    job = manager.get_job(job_id, username, role)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return job


@app.post("/api/jobs")
async def create_jobs(
    request: Request,
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
    username, _ = require_user(request)
    try:
        jobs = manager.create_jobs(
            files=files,
            preset=preset,
            use_gpu=parse_bool(use_gpu),
            keywords_raw=keywords,
            ffmpeg_path=ffmpeg_path,
            owner=username,
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
async def stream_job_events(job_id: str, request: Request) -> StreamingResponse:
    username, role = require_user(request)
    subscriber = manager.subscribe_job(job_id, username, role)
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
async def stream_queue_events(request: Request) -> StreamingResponse:
    require_user(request)
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
async def download_job(job_id: str, request: Request) -> FileResponse:
    username, role = require_user(request)
    info = manager.download_info(job_id, username, role)
    if info is None:
        raise HTTPException(status_code=404, detail="结果文件不存在或任务未完成。")
    path, media_type = info
    return FileResponse(path=path, media_type=media_type, filename=path.name)


@app.post("/api/auth/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> dict:
    user = USERS.get(username.strip())
    if user is None or user["password"] != password:
        raise HTTPException(status_code=401, detail="用户名或密码错误。")
    request.session["username"] = username.strip()
    request.session["role"] = user["role"]
    return {"ok": True, "user": {"username": username.strip(), "role": user["role"]}}


@app.post("/api/auth/logout")
async def logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}


@app.post("/api/auth/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
) -> dict:
    normalized = username.strip()
    if len(normalized) < 3:
        raise HTTPException(status_code=400, detail="用户名至少 3 个字符。")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 个字符。")
    if normalized.lower() == "admin":
        raise HTTPException(status_code=400, detail="该用户名不可注册。")
    with USERS_LOCK:
        if normalized in USERS:
            raise HTTPException(status_code=400, detail="用户名已存在。")
        USERS[normalized] = {"password": password, "role": "user"}
        # 持久化注册用户，密码以 base64 编码落盘。
        save_users_to_file(USERS_FILE, USERS)
    return {"ok": True, "user": {"username": normalized, "role": "user"}}


@app.get("/api/auth/me")
async def auth_me(request: Request) -> dict:
    username, role = require_user(request)
    return {"ok": True, "user": {"username": username, "role": role}}


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=10800, reload=False)


if __name__ == "__main__":
    main()
