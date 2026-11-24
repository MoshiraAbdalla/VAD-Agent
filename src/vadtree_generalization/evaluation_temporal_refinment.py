from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from .agents import parse_reasoning_json
from .eval_agentic_maf import (
    DATASET_DEFAULTS,
    load_temporal_annotations,
    load_video_records,
    make_frame_labels,
    normalize_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VAD-Agent-MIF scores with temporal refinement using "
            "saved per-segment ImageBind features."
        )
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--results-dir", type=Path, required=True, help="VAD-Agent-MIF per-video JSON directory.")
    parser.add_argument(
        "--imagebind-feature-dir",
        type=Path,
        default=Path("vadtree_generalization/imagebind_features"),
        help="Root containing <dataset>/<video_name>/segment_*.npz feature caches.",
    )
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--temporal-annotation-file", type=Path, default=None)
    parser.add_argument("--normal-label", type=int, default=None)
    parser.add_argument("--missing-score", type=float, default=0.0)
    parser.add_argument("--top-ratio", type=float, default=0.15)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--smooth-size", type=int, default=15)
    parser.add_argument("--smooth-sigma", type=float, default=10.0)
    parser.add_argument(
        "--pooling",
        choices=["mean", "selected_mean"],
        default="mean",
        help="How to pool saved frame-level ImageBind features into one segment vector.",
    )
    parser.add_argument(
        "--projection-mode",
        choices=["overlap_aware", "overwrite"],
        default="overlap_aware",
        help="How segment scores are projected to frame scores.",
    )
    parser.add_argument(
        "--include-self-sim",
        action="store_true",
        help="Include each segment as its own neighbor. Default excludes self-similarity.",
    )
    parser.add_argument(
        "--apply-vera-position-weight",
        action="store_true",
        help="Apply video-center Gaussian frame weighting.",
    )
    parser.add_argument("--position-weight-sigma", type=float, default=-1.0)
    parser.add_argument("--strict-features", action="store_true", help="Skip videos with any missing feature row.")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    annotation_file = args.annotation_file or defaults["annotation_file"]
    temporal_annotation_file = args.temporal_annotation_file or defaults["temporal_annotation_file"]
    normal_label = args.normal_label if args.normal_label is not None else int(defaults["normal_label"])

    records = load_video_records(annotation_file)
    temporal = load_temporal_annotations(temporal_annotation_file)
    videos = load_per_video_results(args.results_dir, complete_only=args.complete_only)

    raw_scores_all: list[float] = []
    refined_scores_all: list[float] = []
    labels_all: list[int] = []
    per_video: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    stats = {
        "videos_total": len(videos),
        "videos_used": 0,
        "missing_feature_segments": 0,
        "videos_with_missing_features": 0,
    }

    for result in videos:
        video_name = normalize_name(result.get("video_name", ""))
        if video_name not in records or video_name not in temporal:
            skipped[video_name] = "missing annotations"
            continue

        labels = make_frame_labels(records[video_name], temporal[video_name], normal_label)
        segment_defs, segment_scores = parse_agentic_segments(result)
        if not segment_defs:
            skipped[video_name] = "no scored segments"
            continue

        feature_matrix, missing_features = load_segment_feature_matrix(
            args.imagebind_feature_dir,
            args.dataset,
            video_name,
            segment_defs,
            pooling=args.pooling,
        )
        if missing_features:
            stats["missing_feature_segments"] += len(missing_features)
            stats["videos_with_missing_features"] += 1
            if args.strict_features:
                skipped[video_name] = f"missing feature segments: {missing_features[:5]}"
                continue

        if feature_matrix.shape[0] != len(segment_scores):
            skipped[video_name] = (
                f"feature/score mismatch scores={len(segment_scores)} "
                f"features={feature_matrix.shape[0]}"
            )
            continue

        raw_frame_scores = project_segments_to_frames(
            segment_defs,
            segment_scores,
            len(labels),
            mode=args.projection_mode,
            missing_score=args.missing_score,
        )

        refined_segment_scores = temporal_refine_scores(
            segment_scores=segment_scores,
            feature_matrix=feature_matrix,
            top_ratio=args.top_ratio,
            temp=args.temp,
            include_self=args.include_self_sim,
        )
        refined_segment_scores = gaussian_smooth_1d(
            refined_segment_scores,
            size=args.smooth_size,
            sigma=args.smooth_sigma,
        )
        refined_frame_scores = project_segments_to_frames(
            segment_defs,
            refined_segment_scores,
            len(labels),
            mode=args.projection_mode,
            missing_score=args.missing_score,
        )

        if args.apply_vera_position_weight:
            sigma = args.position_weight_sigma if args.position_weight_sigma > 0 else max(1.0, len(labels) / 2.0)
            refined_frame_scores = gaussian_position_weight(refined_frame_scores, sigma=sigma)

        binary_labels = np.asarray([int(label != normal_label) for label in labels], dtype=np.int32)
        raw_metrics = compute_metrics(raw_frame_scores, binary_labels)
        refined_metrics = compute_metrics(refined_frame_scores, binary_labels)
        per_video[video_name] = {
            "num_segments": len(segment_scores),
            "num_frames": len(labels),
            "positive_frames": int(binary_labels.sum()),
            "missing_feature_segments": missing_features,
            "raw": raw_metrics,
            "temporal_refined": refined_metrics,
        }

        raw_scores_all.extend(raw_frame_scores.tolist())
        refined_scores_all.extend(refined_frame_scores.tolist())
        labels_all.extend(binary_labels.tolist())
        stats["videos_used"] += 1

    payload = {
        "dataset": args.dataset,
        "results_dir": str(args.results_dir),
        "imagebind_feature_dir": str(args.imagebind_feature_dir),
        "annotation_file": str(annotation_file),
        "temporal_annotation_file": str(temporal_annotation_file),
        "normal_label": normal_label,
        "config": {
            "complete_only": args.complete_only,
            "top_ratio": args.top_ratio,
            "temp": args.temp,
            "smooth_size": args.smooth_size,
            "smooth_sigma": args.smooth_sigma,
            "pooling": args.pooling,
            "projection_mode": args.projection_mode,
            "include_self_sim": args.include_self_sim,
            "apply_vera_position_weight": args.apply_vera_position_weight,
            "position_weight_sigma": args.position_weight_sigma,
            "strict_features": args.strict_features,
        },
        "stats": stats,
        "raw_dataset_metrics": compute_metrics(np.asarray(raw_scores_all, dtype=np.float32), np.asarray(labels_all)),
        "temporal_refined_dataset_metrics": compute_metrics(
            np.asarray(refined_scores_all, dtype=np.float32),
            np.asarray(labels_all),
        ),
        "per_video": per_video,
        "skipped": skipped,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "raw": payload["raw_dataset_metrics"],
                "temporal_refined": payload["temporal_refined_dataset_metrics"],
                "stats": payload["stats"],
                "skipped": len(skipped),
            },
            indent=2,
        )
    )


def load_per_video_results(results_dir: Path, complete_only: bool) -> list[dict[str, Any]]:
    videos = []
    for path in sorted(results_dir.glob("*.json")):
        if _is_auxiliary_json(path.name):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "segments" not in data:
            continue
        if complete_only and data.get("complete") is not True:
            continue
        videos.append(data)
    return videos


def _is_auxiliary_json(name: str) -> bool:
    if name == "agentic_results.json":
        return True
    return bool(re.search(r"(eval|vid_score|refined|metrics|deepseek)", name, flags=re.IGNORECASE))


def parse_agentic_segments(result: dict[str, Any]) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    segment_defs: list[tuple[int, int, int]] = []
    scores: list[float] = []
    for segment in result.get("segments", []):
        plan = segment.get("plan", {})
        try:
            segment_id = int(plan.get("segment_id", len(segment_defs)))
            start = int(plan["start"])
            end = int(plan["end"])
        except (KeyError, TypeError, ValueError):
            continue

        score = segment.get("anomaly_score")
        if score is None:
            score, _ = parse_reasoning_json(str(segment.get("reasoning_response") or ""))
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        segment_defs.append((segment_id, start, end))
        scores.append(float(np.clip(score_value, 0.0, 1.0)))
    return segment_defs, np.asarray(scores, dtype=np.float32)


def load_segment_feature_matrix(
    feature_root: Path,
    dataset: str,
    video_name: str,
    segment_defs: list[tuple[int, int, int]],
    pooling: str,
) -> tuple[np.ndarray, list[str]]:
    video_feature_dir = feature_root / dataset / video_name
    rows = []
    missing = []
    for segment_id, start, end in segment_defs:
        pattern = f"segment_{segment_id:06d}_{start}_{end}.npz"
        path = video_feature_dir / pattern
        if not path.exists():
            matches = sorted(video_feature_dir.glob(f"segment_{segment_id:06d}_*.npz"))
            path = matches[0] if matches else path
        if not path.exists():
            missing.append(f"{segment_id}:{start}-{end}")
            rows.append(None)
            continue
        features = np.load(path)["features"].astype(np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            missing.append(f"{segment_id}:{start}-{end}")
            rows.append(None)
            continue
        rows.append(pool_segment_features(features, path, pooling))

    fallback = _fallback_feature(rows)
    out = []
    for row in rows:
        out.append(fallback.copy() if row is None else row)
    if not out:
        return np.empty((0, 0), dtype=np.float32), missing
    return np.stack(out, axis=0).astype(np.float32), missing


def pool_segment_features(features: np.ndarray, npz_path: Path, pooling: str) -> np.ndarray:
    if pooling == "selected_mean":
        meta_path = npz_path.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                indices = [int(i) for i in meta.get("selected_local_indices", [])]
                indices = [i for i in indices if 0 <= i < features.shape[0]]
                if indices:
                    return features[np.asarray(indices, dtype=np.int64)].mean(axis=0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    return features.mean(axis=0)


def _fallback_feature(rows: list[np.ndarray | None]) -> np.ndarray:
    valid = [row for row in rows if row is not None]
    if valid:
        return np.mean(np.stack(valid, axis=0), axis=0).astype(np.float32)
    return np.zeros((1024,), dtype=np.float32)


def temporal_refine_scores(
    segment_scores: np.ndarray,
    feature_matrix: np.ndarray,
    top_ratio: float,
    temp: float,
    include_self: bool,
) -> np.ndarray:
    n = int(segment_scores.size)
    if n == 0:
        return segment_scores.astype(np.float32)
    if n == 1:
        return segment_scores.astype(np.float32)

    feats = feature_matrix.astype(np.float32)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(feats, feats.T)
    if not include_self:
        np.fill_diagonal(sim, -1.0)

    top_n = max(1, int(float(top_ratio) * n))
    if not include_self:
        top_n = min(top_n, max(1, n - 1))
    else:
        top_n = min(top_n, n)

    indices = np.argsort(sim, axis=1)[:, -top_n:]
    values = np.take_along_axis(sim, indices, axis=1)
    refined = []
    for i in range(n):
        weights = softmax(values[i] * float(temp))
        refined.append(float(np.dot(weights, segment_scores[indices[i]])))
    return np.asarray(refined, dtype=np.float32)


def project_segments_to_frames(
    segment_defs: list[tuple[int, int, int]],
    segment_scores: np.ndarray,
    length: int,
    mode: str,
    missing_score: float,
) -> np.ndarray:
    pred = np.full((length,), float(missing_score), dtype=np.float32)
    if mode == "overwrite":
        for (_, start, end), score in zip(segment_defs, segment_scores):
            start = max(0, min(length - 1, int(start)))
            end = max(start, min(length - 1, int(end)))
            pred[start : end + 1] = float(score)
        return pred

    frame_sum = np.zeros((length,), dtype=np.float32)
    frame_count = np.zeros((length,), dtype=np.float32)
    for (_, start, end), score in zip(segment_defs, segment_scores):
        start = max(0, min(length, int(start)))
        end = max(start, min(length - 1, int(end)))
        frame_sum[start : end + 1] += float(score)
        frame_count[start : end + 1] += 1.0
    covered = frame_count > 0
    pred[covered] = frame_sum[covered] / frame_count[covered]
    return pred


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    size = max(1, int(size))
    sigma = max(1e-6, float(sigma))
    xs = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2 * sigma * sigma))
    return kernel / (np.sum(kernel) + 1e-8)


def gaussian_smooth_1d(data: np.ndarray, size: int, sigma: float) -> np.ndarray:
    if data.size == 0:
        return data.astype(np.float32)
    return np.convolve(data.astype(np.float32), gaussian_kernel(size, sigma), mode="same").astype(np.float32)


def gaussian_position_weight(data: np.ndarray, sigma: float) -> np.ndarray:
    n = int(data.size)
    if n == 0:
        return data.astype(np.float32)
    x = np.arange(n, dtype=np.float32)
    mu = float(n // 2)
    weights = np.exp(-0.5 * ((x - mu) / max(1e-6, float(sigma))) ** 2).astype(np.float32)
    return (weights * data.astype(np.float32)).astype(np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    exp = np.exp(values - np.max(values))
    return exp / (np.sum(exp) + 1e-8)


def compute_metrics(scores: np.ndarray, binary_labels: np.ndarray) -> dict[str, Any]:
    if scores.size == 0:
        return {"roc_auc": None, "average_precision": None, "pr_auc": None}
    if len(np.unique(binary_labels)) < 2:
        return {"roc_auc": None, "average_precision": None, "pr_auc": None, "reason": "needs both classes"}
    precision, recall, _ = precision_recall_curve(binary_labels, scores)
    return {
        "roc_auc": round(float(roc_auc_score(binary_labels, scores)), 6),
        "average_precision": round(float(average_precision_score(binary_labels, scores)), 6),
        "pr_auc": round(float(auc(recall, precision)), 6),
    }


if __name__ == "__main__":
    main()

