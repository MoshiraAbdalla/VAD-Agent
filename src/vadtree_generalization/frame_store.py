from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path) -> List[object]:
    """Sort frame paths by numeric chunks, so 000010.jpg follows 000009.jpg."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def resolve_frames_root(dataset_root: Path) -> Path:
    frames_root = dataset_root / "frames"
    return frames_root if frames_root.exists() else dataset_root


def iter_video_dirs(dataset_root: Path, limit: int | None = None) -> Iterable[Path]:
    frames_root = resolve_frames_root(dataset_root)
    video_dirs = [path for path in frames_root.iterdir() if path.is_dir()]
    video_dirs.sort(key=lambda p: p.name.lower())
    if limit is not None:
        video_dirs = video_dirs[:limit]
    yield from video_dirs


def list_frame_paths(video_dir: Path) -> List[Path]:
    frames = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames.sort(key=natural_key)
    return frames


def infer_dataset_name(path_or_name: str) -> str:
    text = path_or_name.lower()
    if "xd" in text or "label_" in text:
        return "xd"
    if "ucf" in text or "_x264" in text:
        return "ucf"
    return "generic"


