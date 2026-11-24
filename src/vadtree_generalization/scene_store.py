from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


SceneMap = Dict[str, List[Tuple[int, int]]]


def _normalize_key(name: str) -> str:
    path = Path(name)
    return path.stem if path.suffix else name


def load_vadtree_scenes(scene_json: Path) -> SceneMap:
    """Load VADTree scene intervals keyed by filename stem.

    VADTree JSON keys usually include `.mp4`, while extracted-frame directories
    often use the stem. Store both forms normalized to the stem.
    """
    with scene_json.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    scenes: SceneMap = {}
    for video_key, record in data.items():
        raw_scenes = record.get("scenes", [])
        intervals = [(int(start), int(end)) for start, end in raw_scenes]
        scenes[_normalize_key(video_key)] = intervals
    return scenes


def get_video_scenes(scene_map: SceneMap, video_dir_name: str) -> List[Tuple[int, int]] | None:
    return scene_map.get(_normalize_key(video_dir_name))


def vadtree_scenes_to_inclusive_ranges(
    scenes: Sequence[Tuple[int, int]],
    num_frames: int,
) -> List[Tuple[int, int]]:
    """Convert VADTree boundary intervals to non-overlapping inclusive ranges.

    VADTree scenes are boundary-style intervals where adjacent scenes often
    share a boundary value, e.g. [0, 577], [577, 580]. For frame-folder slicing,
    convert them to [0, 576], [577, 579], and let the last scene include the
    final available frame.
    """
    if num_frames <= 0:
        return []
    ranges: List[Tuple[int, int]] = []
    for idx, (start, end) in enumerate(scenes):
        start = max(0, min(start, num_frames - 1))
        if idx + 1 < len(scenes):
            next_start = max(0, min(int(scenes[idx + 1][0]), num_frames))
            inclusive_end = min(end - 1, next_start - 1, num_frames - 1)
        else:
            inclusive_end = min(max(end, start), num_frames - 1)

        if inclusive_end < start:
            inclusive_end = start
        ranges.append((start, inclusive_end))
    return ranges


