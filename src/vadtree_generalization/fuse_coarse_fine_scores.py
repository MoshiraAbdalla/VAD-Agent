from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .eval_agentic_maf import (
    DATASET_DEFAULTS,
    load_temporal_annotations,
    load_video_records,
    make_frame_labels,
    normalize_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VADTree-style coarse/fine correlation fusion.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--coarse-json", type=Path, required=True)
    parser.add_argument("--fine-json", type=Path, required=True)
    parser.add_argument("--label", default="fused")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--coarse-score-index", type=int, default=0)
    parser.add_argument("--fine-score-index", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    records = load_video_records(defaults["annotation_file"])
    temporal = load_temporal_annotations(defaults["temporal_annotation_file"])
    normal_label = int(defaults["normal_label"])

    coarse_payload = _read_json(args.coarse_json)
    fine_payload = _read_json(args.fine_json)
    coarse_vid = coarse_payload.get("vid_score", {})
    fine_vid = fine_payload.get("vid_score", {})

    labels_all: list[int] = []
    scores_all: list[float] = []
    fused_payload: dict[str, Any] = {
        "config": {
            "method": "vadtree_correlation_fusion",
            "beta": args.beta,
            "coarse_json": str(args.coarse_json),
            "fine_json": str(args.fine_json),
            "coarse_score_index": args.coarse_score_index,
            "fine_score_index": args.fine_score_index,
        },
        "vid_score": {},
    }
    skipped: dict[str, str] = {}
    used = 0

    common_keys = sorted(set(coarse_vid) & set(fine_vid))
    for video_key in common_keys:
        name = normalize_name(video_key)
        if name not in records or name not in temporal:
            skipped[video_key] = "missing annotations"
            continue
        labels = make_frame_labels(records[name], temporal[name], normal_label)
        y = [int(label != normal_label) for label in labels]
        fused = fuse_video(
            coarse_vid[video_key],
            fine_vid[video_key],
            len(labels),
            beta=args.beta,
            coarse_score_index=args.coarse_score_index,
            fine_score_index=args.fine_score_index,
        )
        labels_all.extend(y)
        scores_all.extend(fused)
        fused_payload["vid_score"][video_key] = [round(float(score), 4) for score in fused]
        used += 1

    metrics = {
        "label": args.label,
        "videos": used,
        "frames": len(labels_all),
        "positive_frames": int(sum(labels_all)),
        "roc_auc": float(roc_auc_score(labels_all, scores_all)),
        "average_precision": float(average_precision_score(labels_all, scores_all)),
        "skipped": skipped,
    }
    fused_payload["dataset_metric"] = metrics
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(fused_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{args.label}\tvideos={metrics['videos']}\tframes={metrics['frames']}"
        f"\tpos={metrics['positive_frames']}\tauc={metrics['roc_auc']:.6f}"
        f"\tap={metrics['average_precision']:.6f}\tskipped={len(skipped)}"
    )


def fuse_video(
    coarse_scores: dict[str, Any],
    fine_scores: dict[str, Any],
    num_frames: int,
    beta: float,
    coarse_score_index: int,
    fine_score_index: int,
) -> list[float]:
    coarse_frame_scores = _project_to_frames(coarse_scores, num_frames, coarse_score_index)
    fine_frame_scores = _project_to_frames(fine_scores, num_frames, fine_score_index)
    norm_variances = _normalized_fine_variances(coarse_scores, fine_scores, coarse_score_index, fine_score_index)

    weights = np.full((num_frames,), 0.5, dtype=np.float32)
    for key in coarse_scores:
        parsed = _parse_edge(key)
        if parsed is None:
            continue
        start, end = parsed
        start = max(0, start)
        end = min(num_frames - 1, end)
        if end < start:
            continue
        fine_w = norm_variances.get(key, 0.0) * beta
        weights[start : end + 1] = 0.5 - fine_w / 2.0

    fused = weights * coarse_frame_scores + (1.0 - weights) * fine_frame_scores
    return [float(score) for score in fused]


def _normalized_fine_variances(
    coarse_scores: dict[str, Any],
    fine_scores: dict[str, Any],
    coarse_score_index: int,
    fine_score_index: int,
) -> dict[str, float]:
    variances: dict[str, float] = {}
    fine_items = []
    for key, value in fine_scores.items():
        edge = _parse_edge(key)
        if edge is not None:
            fine_items.append((key, edge, _extract_score(value, fine_score_index)))

    for coarse_key, coarse_value in coarse_scores.items():
        coarse_edge = _parse_edge(coarse_key)
        if coarse_edge is None:
            continue
        c_start, c_end = coarse_edge
        matched = [
            score
            for _, (f_start, f_end), score in fine_items
            if f_start >= c_start and f_end <= c_end and score is not None
        ]
        if matched:
            variances[coarse_key] = float(np.var(np.asarray(matched, dtype=np.float32)))
        else:
            variances[coarse_key] = 0.0

    values = np.asarray(list(variances.values()), dtype=np.float32)
    if values.size == 0:
        return {}
    min_val = float(values.min())
    max_val = float(values.max())
    denom = max_val - min_val + 1e-8
    return {key: float((value - min_val) / denom) for key, value in variances.items()}


def _project_to_frames(scores: dict[str, Any], num_frames: int, score_index: int) -> np.ndarray:
    out = np.zeros((num_frames,), dtype=np.float32)
    for key, value in scores.items():
        edge = _parse_edge(key)
        if edge is None:
            continue
        score = _extract_score(value, score_index)
        if score is None or score < 0:
            continue
        start, end = edge
        start = max(0, start)
        end = min(num_frames - 1, end)
        if end >= start:
            out[start : end + 1] = float(score)
    return out


def _extract_score(value: Any, score_index: int) -> float | None:
    if isinstance(value, list):
        if score_index >= len(value):
            return None
        value = value[score_index]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_edge(edge: str) -> tuple[int, int] | None:
    parts = [part.strip() for part in str(edge).split(",")]
    if len(parts) != 2:
        return None
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

