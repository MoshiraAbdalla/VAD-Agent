from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Tuple

import numpy as np

from .schemas import FrameRef, SegmentPlan


class FrameSelector(Protocol):
    def select(
        self,
        frame_paths: Sequence[Path],
        num_samples: int,
        cache_info: Dict[str, Any] | None = None,
    ) -> Tuple[List[int], Dict[int, float]]:
        ...


class UniformFrameSelector:
    def select(
        self,
        frame_paths: Sequence[Path],
        num_samples: int,
        cache_info: Dict[str, Any] | None = None,
    ) -> Tuple[List[int], Dict[int, float]]:
        selected = uniform_indices(len(frame_paths), num_samples)
        return selected, {idx: 1.0 for idx in selected}


def uniform_indices(length: int, num_samples: int) -> List[int]:
    if length <= 0:
        return []
    if num_samples >= length:
        return list(range(length))
    return np.linspace(0, length - 1, num_samples, dtype=int).tolist()


def segment_ranges(num_frames: int, segment_size: int, stride: int | None = None) -> List[Tuple[int, int]]:
    if num_frames <= 0:
        return []
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")
    stride = segment_size if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")

    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + segment_size - 1, num_frames - 1)
        ranges.append((start, end))
        if end == num_frames - 1:
            break
        start += stride
    return ranges


def select_feature_local_minima(
    features: np.ndarray,
    num_samples: int,
    window: int = -1,
    alpha: float = 2.5,
) -> Tuple[List[int], Dict[int, float]]:
    """Select low local-similarity frames from precomputed visual features.

    This follows the MIF-style local-minima idea, but computes only local
    window similarities instead of materializing an NxN similarity matrix.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got shape {features.shape}")

    n = int(features.shape[0])
    if n == 0 or num_samples <= 0:
        return [], {}
    if n <= num_samples:
        return list(range(n)), {i: 1.0 for i in range(n)}
    if n < 3:
        selected = uniform_indices(n, num_samples)
        return selected, {i: 1.0 for i in selected}

    feats = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-8)

    if window == -1:
        window = max(1, int(n // (alpha * max(num_samples, 1))))
    else:
        window = max(1, int(window))
    if (2 * window) >= n:
        window = max(1, (n - 1) // 2)

    local_avg = np.full((n,), np.inf, dtype=np.float32)
    for idx in range(window, n - window):
        left = idx - window
        right = idx + window + 1
        sims = feats[left:right] @ feats[idx]
        local_avg[idx] = float((sims.sum() - 1.0) / max(len(sims) - 1, 1))

    valid = np.flatnonzero(np.isfinite(local_avg))
    if len(valid) == 0:
        selected = uniform_indices(n, num_samples)
        return selected, {i: 1.0 for i in selected}

    selected: List[int] = [int(valid[np.argmin(local_avg[valid])])]
    intervals: List[Tuple[float, int, int, int]] = []
    _push_best_interval(intervals, local_avg, 0, selected[0] - window + 1)
    _push_best_interval(intervals, local_avg, selected[0] + window, n)

    while len(selected) < num_samples and intervals:
        _, start, end, idx = heapq.heappop(intervals)
        selected.append(idx)
        _push_best_interval(intervals, local_avg, start, idx - window + 1)
        _push_best_interval(intervals, local_avg, idx + window, end)

    if len(selected) < num_samples:
        ranked = sorted(valid.tolist(), key=lambda i: float(local_avg[i]))
        for idx in ranked:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) == num_samples:
                break

    if len(selected) < num_samples:
        for idx in uniform_indices(n, num_samples):
            if idx not in selected:
                selected.append(idx)
            if len(selected) == num_samples:
                break

    selected = sorted(set(selected))
    scores = {idx: round(float(1.0 - local_avg[idx]), 6) for idx in selected if np.isfinite(local_avg[idx])}
    for idx in selected:
        scores.setdefault(idx, 1.0)
    return selected, scores


def _push_best_interval(
    heap: List[Tuple[float, int, int, int]],
    scores: np.ndarray,
    start: int,
    end: int,
) -> None:
    if end <= start:
        return
    candidates = np.arange(start, end, dtype=np.int64)
    candidates = candidates[np.isfinite(scores[candidates])]
    if len(candidates) == 0:
        return
    idx = int(candidates[np.argmin(scores[candidates])])
    heapq.heappush(heap, (float(scores[idx]), int(start), int(end), idx))


def build_segment_plans(
    frame_paths: Sequence[Path],
    segment_size: int,
    context_samples: int,
    reasoning_samples: int,
    frame_selector: FrameSelector,
    stride: int | None = None,
    ranges: Sequence[Tuple[int, int]] | None = None,
) -> List[SegmentPlan]:
    plans: List[SegmentPlan] = []
    segment_bounds = list(ranges) if ranges is not None else segment_ranges(len(frame_paths), segment_size, stride)
    for segment_id, (start, end) in enumerate(segment_bounds):
        start = max(0, min(int(start), len(frame_paths) - 1))
        end = max(start, min(int(end), len(frame_paths) - 1))
        segment_paths = list(frame_paths[start : end + 1])
        context_local = uniform_indices(len(segment_paths), context_samples)
        selected_local, selection_scores_local = frame_selector.select(
            segment_paths,
            reasoning_samples,
            cache_info={
                "segment_id": segment_id,
                "start": start,
                "end": end,
                "global_frame_indices": list(range(start, end + 1)),
            },
        )

        context_frames = [
            FrameRef(index=start + local_idx, path=segment_paths[local_idx])
            for local_idx in context_local
        ]
        selected_frames = [
            FrameRef(index=start + local_idx, path=segment_paths[local_idx])
            for local_idx in selected_local
        ]
        selection_scores = {start + idx: score for idx, score in selection_scores_local.items()}
        plans.append(
            SegmentPlan(
                segment_id=segment_id,
                start=start,
                end=end,
                context_frames=context_frames,
                selected_frames=selected_frames,
                selection_scores=selection_scores,
            )
        )
    return plans

