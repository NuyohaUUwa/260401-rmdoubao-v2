#!/usr/bin/env python3
"""
Watermark removal core.

Pipeline: EasyOCR per frame + OpenCV inpaint, then mux with original audio (ffmpeg).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ProgressCallback = Callable[[int, int], None]
StageCallback = Callable[[str, str], None]


@dataclass
class Detection:
    boxes: list[tuple[int, int, int, int]]
    merged: tuple[int, int, int, int]
    corner: str
    instances: list[tuple[int, int, int, int]]
    mode: str = "rect"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def work_temp_dir() -> Path:
    p = repo_root() / "output" / "_tmp_frames"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_stem(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return s or "video"


def cleanup_tmpdir(tmpdir: Path) -> None:
    for _ in range(40):
        if not tmpdir.exists():
            break
        for p in sorted(tmpdir.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    p.rmdir()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass
        if not tmpdir.exists():
            break
        shutil.rmtree(tmpdir, ignore_errors=True)
        if not tmpdir.exists():
            break
        time.sleep(0.5)
    if tmpdir.exists() and os.name == "nt":
        subprocess.run(
            ["cmd", "/d", "/c", "rmdir", "/s", "/q", str(tmpdir)],
            capture_output=True,
            text=True,
        )
    root = work_temp_dir()
    try:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    except OSError:
        pass


def resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"ffmpeg not found: {explicit}")
    env = os.environ.get("DOUBAO_FFMPEG")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    bundled = repo_root() / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    which = shutil.which("ffmpeg")
    if which:
        return which
    raise FileNotFoundError(
        "未找到 ffmpeg。请设置环境变量 DOUBAO_FFMPEG 或将 ffmpeg.exe 放在项目根目录。"
    )


def probe_video(video_path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"无法读取分辨率: {video_path}")
    return fps, w, h


def probe_video_with_frames(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"无法读取分辨率: {video_path}")
    return fps, w, h, frames


def validate_output_video(input_path: Path, output_path: Path) -> None:
    in_fps, in_w, in_h, in_frames = probe_video_with_frames(input_path)
    out_fps, out_w, out_h, out_frames = probe_video_with_frames(output_path)
    problems: list[str] = []
    if out_w != in_w or out_h != in_h:
        problems.append(f"输出分辨率异常: {out_w}x{out_h}，应为 {in_w}x{in_h}")
    if in_frames > 0 and out_frames > 0 and abs(out_frames - in_frames) > 1:
        problems.append(f"输出帧数异常: {out_frames}，应接近 {in_frames}")
    if abs(out_fps - in_fps) > 0.1:
        problems.append(f"输出帧率异常: {out_fps:.3f}，应接近 {in_fps:.3f}")
    if problems:
        raise RuntimeError("输出视频校验失败；" + "；".join(problems))


def run_ffmpeg(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.stderr.write(r.stderr or r.stdout or "")
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)


def has_audio_stream(ffmpeg: str, video_path: Path) -> bool:
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path.resolve())],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    err = r.stderr or ""
    return "Audio:" in err or ("Stream #" in err and "Audio" in err)


def bbox_from_easyocr_box(box: np.ndarray) -> tuple[int, int, int, int]:
    xs = box[:, 0]
    ys = box[:, 1]
    x0, y0 = int(np.floor(xs.min())), int(np.floor(ys.min()))
    x1, y1 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    return x0, y0, x1, y1


def expand_bbox(
    x0: int, y0: int, x1: int, y1: int, pad: int, w: int, h: int
) -> tuple[int, int, int, int]:
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def text_matches(text: str, patterns: list[re.Pattern[str]]) -> bool:
    t = text.replace(" ", "").strip()
    for p in patterns:
        if p.search(t):
            return True
    return False


def infer_torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_use_gpu(requested: bool | None) -> bool:
    if requested is False:
        return False
    if requested is True:
        return infer_torch_cuda()
    return infer_torch_cuda()


def normalize_ocr_text(text: str) -> str:
    t = text.replace(" ", "").strip()
    t = t.upper()
    return (
        t.replace("A1", "AI")
        .replace("AI1", "AI")
        .replace("豆A1生成", "豆包AI生成")
        .replace("豆AI生成", "豆包AI生成")
    )


def fuzzy_watermark_match(text: str) -> bool:
    t = normalize_ocr_text(text)
    if not t:
        return False
    target = "豆包AI生成"
    if "豆" in t and any(ch in t for ch in ("包", "AI", "A", "I", "1", "生", "成", "尿", "0", "4")):
        return True
    return SequenceMatcher(None, t, target).ratio() >= 0.45


def _ocr_boxes_from_results(
    results: list[Any],
    patterns: list[re.Pattern[str]],
    pad: int,
    fw: int,
    fh: int,
    ox: int = 0,
    oy: int = 0,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for item in results:
        if len(item) < 2:
            continue
        box, text = item[0], item[1]
        txt = str(text)
        if not text_matches(txt, patterns) and not fuzzy_watermark_match(txt):
            continue
        arr = np.array(box, dtype=np.float32)
        x0, y0, x1, y1 = bbox_from_easyocr_box(arr)
        x0, y0, x1, y1 = x0 + ox, y0 + oy, x1 + ox, y1 + oy
        boxes.append(expand_bbox(x0, y0, x1, y1, pad, fw, fh))
    return boxes


def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


def classify_corner(box: tuple[int, int, int, int], w: int, h: int) -> str:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    if cy < h * 0.38:
        return "tr"
    if cx < w * 0.45:
        return "bl"
    return "br"


def preprocess_ocr_variants(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th180 = cv2.threshold(up, 180, 255, cv2.THRESH_BINARY)
    _, th160 = cv2.threshold(up, 160, 255, cv2.THRESH_BINARY)
    return [image, gray, up, th180, th160]


def corner_rois(w: int, h: int) -> list[tuple[int, int, int, int]]:
    return [
        (0, int(h * 0.28), int(w * 0.42), h),
        (int(w * 0.56), int(h * 0.56), w, h),
        (int(w * 0.66), 0, w, int(h * 0.34)),
    ]


def _make_detection(
    boxes: list[tuple[int, int, int, int]],
    w: int,
    h: int,
    *,
    instances: list[tuple[int, int, int, int]] | None = None,
    mode: str = "rect",
) -> Detection | None:
    merged = _merge_boxes(boxes)
    if merged is None:
        return None
    if instances is None:
        instances = boxes[:]
    return Detection(boxes=boxes, merged=merged, corner=classify_corner(merged, w, h), instances=instances, mode=mode)


def _split_box_by_text_count(
    box: tuple[int, int, int, int],
    text: str,
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    compact = text.replace(" ", "").strip()
    n = max(1, min(len(compact), 20))
    # 近似切分字符框：横排按宽度切分，竖排按高度切分。
    horizontal = w >= h
    chunks: list[tuple[int, int, int, int]] = []
    if horizontal:
        step = w / n
        for i in range(n):
            cx0 = int(round(x0 + i * step))
            cx1 = int(round(x0 + (i + 1) * step))
            if cx1 <= cx0:
                continue
            chunks.append((cx0, y0, cx1, y1))
    else:
        step = h / n
        for i in range(n):
            cy0 = int(round(y0 + i * step))
            cy1 = int(round(y0 + (i + 1) * step))
            if cy1 <= cy0:
                continue
            chunks.append((x0, cy0, x1, cy1))
    return chunks or [box]


def _segment_text_instances(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    min_instance_area: int,
) -> list[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = expand_bbox(x0, y0, x1, y1, 1, w, h)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return [box]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [box]
    instances: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area < min_instance_area or rw < 2 or rh < 4:
            continue
        ex0 = max(0, x0 + rx - 1)
        ey0 = max(0, y0 + ry - 1)
        ex1 = min(w, x0 + rx + rw + 1)
        ey1 = min(h, y0 + ry + rh + 1)
        if ex1 > ex0 and ey1 > ey0:
            instances.append((ex0, ey0, ex1, ey1))
    if not instances:
        return [box]
    instances.sort(key=lambda b: (b[0], b[1]))
    return instances[:24]


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]], *, threshold: float = 0.75) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]))):
        if any(iou(box, old) >= threshold for old in kept):
            continue
        kept.append(box)
    return kept


def _ocr_instances_from_results(
    frame: np.ndarray,
    results: list[Any],
    patterns: list[re.Pattern[str]],
    pad: int,
    fw: int,
    fh: int,
    *,
    ox: int = 0,
    oy: int = 0,
    min_instance_area: int = 12,
) -> list[tuple[int, int, int, int]]:
    instances: list[tuple[int, int, int, int]] = []
    for item in results:
        if len(item) < 2:
            continue
        box, text = item[0], str(item[1])
        if not text_matches(text, patterns) and not fuzzy_watermark_match(text):
            continue
        arr = np.array(box, dtype=np.float32)
        bx0, by0, bx1, by1 = bbox_from_easyocr_box(arr)
        bx0, by0, bx1, by1 = bx0 + ox, by0 + oy, bx1 + ox, by1 + oy
        coarse = _split_box_by_text_count((bx0, by0, bx1, by1), text)
        for cbox in coarse:
            sx0, sy0, sx1, sy1 = expand_bbox(*cbox, pad // 3, fw, fh)
            segmented = _segment_text_instances(
                frame,
                (sx0, sy0, sx1, sy1),
                min_instance_area=max(4, min_instance_area),
            )
            for inst in segmented:
                instances.append(expand_bbox(*inst, 1, fw, fh))
    return _dedupe_boxes(instances)


def readtext_keyword_instances(
    reader: Any,
    frame: np.ndarray,
    image: np.ndarray,
    patterns: list[re.Pattern[str]],
    pad: int,
    fw: int,
    fh: int,
    ox: int = 0,
    oy: int = 0,
    *,
    min_instance_area: int = 12,
    mag_ratio: float = 1.0,
    text_threshold: float = 0.45,
    low_text: float = 0.35,
    link_threshold: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    try:
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            mag_ratio=mag_ratio,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
        )
    except Exception:
        return []
    return _ocr_instances_from_results(
        frame=frame,
        results=results,
        patterns=patterns,
        pad=pad,
        fw=fw,
        fh=fh,
        ox=ox,
        oy=oy,
        min_instance_area=min_instance_area,
    )


def detect_watermark_boxes(
    reader: Any,
    frame: np.ndarray,
    patterns: list[re.Pattern[str]],
    pad: int,
    *,
    min_instance_area: int = 12,
) -> Detection | None:
    h, w = frame.shape[:2]
    char_direct = readtext_keyword_instances(
        reader,
        frame,
        frame,
        patterns,
        pad,
        w,
        h,
        min_instance_area=min_instance_area,
    )
    if char_direct:
        return _make_detection(char_direct, w, h, instances=char_direct, mode="char")

    char_found: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in corner_rois(w, h):
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        for variant in preprocess_ocr_variants(roi):
            instances = readtext_keyword_instances(
                reader,
                frame,
                variant,
                patterns,
                pad,
                w,
                h,
                x0,
                y0,
                min_instance_area=min_instance_area,
                mag_ratio=1.25,
                text_threshold=0.30,
                low_text=0.20,
                link_threshold=0.20,
            )
            if instances:
                char_found.extend(instances)
                break
    if char_found:
        char_found = _dedupe_boxes(char_found)
        return _make_detection(char_found, w, h, instances=char_found, mode="char")

    direct = readtext_keyword_boxes(reader, frame, patterns, pad, w, h)
    if direct is not None:
        return Detection(boxes=[direct], merged=direct, corner=classify_corner(direct, w, h), instances=[direct], mode="rect")

    found: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in corner_rois(w, h):
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        for variant in preprocess_ocr_variants(roi):
            box = readtext_keyword_boxes(
                reader,
                variant,
                patterns,
                pad,
                w,
                h,
                x0,
                y0,
                mag_ratio=1.25,
                text_threshold=0.30,
                low_text=0.20,
                link_threshold=0.20,
            )
            if box is not None:
                found.append(box)
                break
    det = _make_detection(found, w, h, instances=found, mode="rect")
    if det is not None:
        return det

    found = []
    for x0, y0, x1, y1 in corner_rois(w, h):
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        box = readtext_keyword_boxes(
            reader,
            roi,
            patterns,
            pad,
            w,
            h,
            x0,
            y0,
            mag_ratio=1.35,
            text_threshold=0.38,
            low_text=0.30,
            link_threshold=0.32,
        )
        if box is not None:
            found.append(box)
    return _make_detection(found, w, h, instances=found, mode="rect")


def refine_detections(
    detections: list[Detection | None],
    total: int,
    *,
    support_window: int = 3,
    gap_window: int = 8,
) -> list[Detection | None]:
    supported: list[Detection | None] = [None] * total

    for i, det in enumerate(detections):
        if det is None:
            continue
        ok = False
        for j in range(max(0, i - support_window), min(total, i + support_window + 1)):
            if j == i or detections[j] is None:
                continue
            if detections[j].corner == det.corner:
                ok = True
                break
        supported[i] = det if ok else None

    resolved = supported[:]
    for i, det in enumerate(resolved):
        if det is not None:
            continue

        prev_idx: int | None = None
        next_idx: int | None = None
        for j in range(i - 1, max(-1, i - gap_window - 1), -1):
            if j >= 0 and supported[j] is not None:
                prev_idx = j
                break
        for j in range(i + 1, min(total, i + gap_window + 1)):
            if supported[j] is not None:
                next_idx = j
                break

        if prev_idx is None and next_idx is None:
            continue
        if prev_idx is None:
            resolved[i] = supported[next_idx]
            continue
        if next_idx is None:
            resolved[i] = supported[prev_idx]
            continue

        prev_det = supported[prev_idx]
        next_det = supported[next_idx]
        if prev_det is None or next_det is None:
            continue
        if prev_det.corner == next_det.corner:
            resolved[i] = prev_det if (i - prev_idx) <= (next_idx - i) else next_det
        else:
            resolved[i] = next_det if (next_idx - i) <= (i - prev_idx) else prev_det

    return resolved


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    area_a = float(max(1, ax1 - ax0) * max(1, ay1 - ay0))
    area_b = float(max(1, bx1 - bx0) * max(1, by1 - by0))
    return inter / max(1.0, area_a + area_b - inter)


def smooth_box_sequence(
    detections: list[Detection | None],
    total: int,
    *,
    window: int = 2,
) -> list[Detection | None]:
    smoothed = detections[:]
    for i, det in enumerate(detections):
        if det is None:
            continue
        neighbors: list[tuple[int, int, int, int]] = []
        for j in range(max(0, i - window), min(total, i + window + 1)):
            other = detections[j]
            if other is None or other.corner != det.corner:
                continue
            if i != j and iou(det.merged, other.merged) < 0.35:
                continue
            neighbors.append(other.merged)
        if not neighbors:
            continue
        arr = np.array(neighbors, dtype=np.int32)
        merged = tuple(int(v) for v in np.median(arr, axis=0).tolist())
        smoothed[i] = Detection(
            boxes=det.boxes[:],
            merged=merged,
            corner=det.corner,
            instances=det.instances[:],
            mode=det.mode,
        )
    return smoothed


def enforce_local_consistency(
    detections: list[Detection | None],
    total: int,
) -> list[Detection | None]:
    fixed = detections[:]
    for i in range(1, total - 1):
        prev_det = fixed[i - 1]
        curr_det = fixed[i]
        next_det = fixed[i + 1]
        if prev_det is None or next_det is None:
            continue
        if prev_det.corner != next_det.corner:
            continue

        prev_next_iou = iou(prev_det.merged, next_det.merged)
        if prev_next_iou < 0.65:
            continue

        stable_box = tuple(
            int(v)
            for v in np.median(
                np.array([prev_det.merged, next_det.merged], dtype=np.int32), axis=0
            ).tolist()
        )

        if curr_det is None:
            fixed[i] = Detection(
                boxes=[stable_box],
                merged=stable_box,
                corner=prev_det.corner,
                instances=prev_det.instances[:] if prev_det.instances else [stable_box],
                mode=prev_det.mode,
            )
            continue

        if curr_det.corner != prev_det.corner:
            fixed[i] = Detection(
                boxes=[stable_box],
                merged=stable_box,
                corner=prev_det.corner,
                instances=prev_det.instances[:] if prev_det.instances else [stable_box],
                mode=prev_det.mode,
            )
            continue

        if iou(curr_det.merged, prev_det.merged) < 0.35 and iou(curr_det.merged, next_det.merged) < 0.35:
            fixed[i] = Detection(
                boxes=[stable_box],
                merged=stable_box,
                corner=prev_det.corner,
                instances=prev_det.instances[:] if prev_det.instances else [stable_box],
                mode=prev_det.mode,
            )
    return fixed


def instances_to_mask(
    frame: np.ndarray,
    det: Detection,
    *,
    mask_mode: str = "char",
    char_dilate: int = 1,
    char_blur: int = 3,
) -> np.ndarray:
    if mask_mode != "char" or det.mode != "char" or not det.instances:
        return box_to_mask(frame, det.boxes)
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in det.instances:
        x0, y0, x1, y1 = expand_bbox(x0, y0, x1, y1, max(0, char_dilate), w, h)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    k = max(1, int(char_blur))
    if k % 2 == 0:
        k += 1
    if k > 1:
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    _, mask = cv2.threshold(mask, 8, 255, cv2.THRESH_BINARY)
    return mask


def box_to_mask(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        px = max(2, int(round(min(bw, bh) * 0.12)))
        py = px
        ex0, ey0, ex1, ey1 = expand_bbox(x0, y0, x1, y1, px, w, h)
        cv2.rectangle(mask, (ex0, ey0), (ex1, ey1), 255, thickness=-1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 8, 255, cv2.THRESH_BINARY)
    return mask


def choose_inpaint_radius(boxes: list[tuple[int, int, int, int]], fallback: int) -> int:
    if not boxes:
        return fallback
    heights = [max(1, y1 - y0) for _, y0, _, y1 in boxes]
    median_h = float(np.median(heights))
    return max(2, min(fallback, int(round(median_h * 0.10)) + 1))


def region_motion_score(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = box
    prev_roi = prev_frame[y0:y1, x0:x1]
    curr_roi = curr_frame[y0:y1, x0:x1]
    if prev_roi.size == 0 or curr_roi.size == 0:
        return 255.0
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray)))


def temporal_blend_repair(
    repaired: np.ndarray,
    current_frame: np.ndarray,
    prev_repaired: np.ndarray | None,
    prev_frame: np.ndarray | None,
    det: Detection | None,
    prev_det: Detection | None,
    *,
    mask_mode: str = "char",
    char_dilate: int = 1,
    char_blur: int = 3,
) -> np.ndarray:
    if (
        prev_repaired is None
        or prev_frame is None
        or det is None
        or prev_det is None
        or det.corner != prev_det.corner
        or iou(det.merged, prev_det.merged) < 0.55
    ):
        return repaired

    motion = region_motion_score(prev_frame, current_frame, det.merged)
    if motion > 8.0:
        return repaired

    mask = instances_to_mask(
        current_frame,
        det,
        mask_mode=mask_mode,
        char_dilate=char_dilate,
        char_blur=max(5, char_blur),
    )
    blur = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
    if float(blur.max()) <= 0.01:
        return repaired

    alpha = blur[..., None] * 0.35
    blended = repaired.astype(np.float32) * (1.0 - alpha) + prev_repaired.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def readtext_keyword_boxes(
    reader: Any,
    image: np.ndarray,
    patterns: list[re.Pattern[str]],
    pad: int,
    fw: int,
    fh: int,
    ox: int = 0,
    oy: int = 0,
    *,
    mag_ratio: float = 1.0,
    text_threshold: float = 0.45,
    low_text: float = 0.35,
    link_threshold: float = 0.35,
) -> tuple[int, int, int, int] | None:
    try:
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            mag_ratio=mag_ratio,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
        )
    except Exception:
        return None
    boxes = _ocr_boxes_from_results(results, patterns, pad, fw, fh, ox, oy)
    return _merge_boxes(boxes)


def mode_auto(
    input_path: Path,
    output_path: Path,
    ffmpeg: str,
    keywords: list[str],
    pad: int,
    inpaint_radius: int,
    crf: int,
    ocr_interval: int,
    carry_bbox: bool,
    progress: ProgressCallback | None = None,
    scan_progress: ProgressCallback | None = None,
    use_gpu: bool | None = None,
    roi_fallback: bool = True,
    stage_callback: StageCallback | None = None,
    mask_mode: str = "char",
    char_dilate: int = 1,
    char_blur: int = 3,
    track_gap: int = 8,
    min_instance_area: int = 12,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> None:
    del ocr_interval
    del carry_bbox
    del roi_fallback
    try:
        import easyocr
    except ImportError as e:
        raise SystemExit("请先安装 easyocr: pip install easyocr") from e

    if stage_callback is not None:
        stage_callback("initializing_ocr", "初始化 OCR")

    gpu = resolve_use_gpu(use_gpu)
    fps, _, _ = probe_video(input_path)
    patterns = [re.compile(re.escape(k)) for k in keywords]

    reader = easyocr.Reader(["ch_sim", "en"], gpu=gpu, verbose=False)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {input_path}")

    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    detections: list[Detection | None] = []

    if stage_callback is not None:
        stage_callback("scanning", "扫描水印位置")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detect_watermark_boxes(reader, frame, patterns, pad, min_instance_area=min_instance_area)
        detections.append(det)
        if scan_progress is not None:
            idx = len(detections)
            tot = total_est if total_est > 0 else idx
            if idx == 1 or idx % 5 == 0 or (total_est > 0 and idx == total_est):
                scan_progress(idx, tot)
    cap.release()

    if not detections:
        raise SystemExit("未读取到任何帧")

    resolved = refine_detections(detections, len(detections), gap_window=max(2, track_gap))
    resolved = smooth_box_sequence(resolved, len(resolved))
    resolved = enforce_local_consistency(resolved, len(resolved))

    tmpdir = work_temp_dir() / f"doubao_wm_{safe_stem(input_path.stem)}"
    if tmpdir.exists():
        cleanup_tmpdir(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"无法重新打开视频: {input_path}")
    try:
        if stage_callback is not None:
            stage_callback("repairing", "修复视频帧")

        frame_idx = 0
        prev_input_frame: np.ndarray | None = None
        prev_output_frame: np.ndarray | None = None
        prev_det: Detection | None = None
        fallback_rect_count = 0
        repaired_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            det = resolved[frame_idx] if frame_idx < len(resolved) else None
            curr_input = frame.copy()
            in_range = frame_start is None or frame_end is None or frame_start <= frame_idx <= frame_end
            should_repair = det is not None and in_range
            if should_repair:
                mask = instances_to_mask(
                    frame,
                    det,
                    mask_mode=mask_mode,
                    char_dilate=max(0, char_dilate),
                    char_blur=max(1, char_blur),
                )
                radius = choose_inpaint_radius(det.boxes, inpaint_radius)
                frame = cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)
                frame = temporal_blend_repair(
                    frame,
                    curr_input,
                    prev_output_frame,
                    prev_input_frame,
                    det,
                    prev_det,
                    mask_mode=mask_mode,
                    char_dilate=char_dilate,
                    char_blur=char_blur,
                )
                repaired_count += 1
                if mask_mode == "char" and det.mode != "char":
                    fallback_rect_count += 1
            elif not in_range:
                prev_input_frame = None
                prev_output_frame = None
                prev_det = None
            out_png = tmpdir / f"frame_{frame_idx:06d}.png"
            if not cv2.imwrite(str(out_png), frame):
                raise RuntimeError(f"写入中间帧失败: {out_png}")
            if in_range:
                prev_input_frame = curr_input
                prev_output_frame = frame.copy()
                prev_det = det
            frame_idx += 1
            if progress is not None:
                tot = total_est if total_est > 0 else max(frame_idx, 1)
                progress(frame_idx, tot)
        if mask_mode == "char" and repaired_count > 0 and fallback_rect_count > 0:
            print(
                f"[watermark] 字符级回退到矩形模式: {fallback_rect_count}/{repaired_count} 帧",
                file=sys.stderr,
            )

        cap.release()
        if frame_idx == 0:
            raise SystemExit("未读取到任何帧")

        if stage_callback is not None:
            stage_callback("muxing", "封装输出视频")

        out_path = output_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(tmpdir / "frame_%06d.png")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-i",
            str(input_path.resolve()),
            "-map",
            "0:v",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
        ]
        if has_audio_stream(ffmpeg, input_path):
            cmd.extend(["-map", "1:a:0", "-c:a", "copy"])
        cmd.extend(["-shortest", str(out_path)])
        run_ffmpeg(cmd)

        if stage_callback is not None:
            stage_callback("validating", "校验输出")

        validate_output_video(input_path, out_path)
    finally:
        if cap.isOpened():
            cap.release()
        cleanup_tmpdir(tmpdir)
