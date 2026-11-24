from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from .agents import parse_reasoning_json


DATASET_DEFAULTS = {
    "ucf": {
        "annotation_file": Path("dataset_info/ucf_crime/annotations/anomaly_test.txt"),
        "temporal_annotation_file": Path("dataset_info/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
        "normal_label": 7,
    },
    "xd": {
        "annotation_file": Path("dataset_info/xd_violence/annotations/anomaly_test.txt"),
        "temporal_annotation_file": Path("dataset_info/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt"),
        "normal_label": 4,
    },
    "msad": {
        "annotation_file": Path("dataset_info/MSAD/annotations/anomaly_test.txt"),
        "temporal_annotation_file": Path("dataset_info/MSAD/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
        "normal_label": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VAD-Agent-MIF segment scores with frame-level ROC AUC/AP.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to agentic_results.json, a per-video JSON, or an output directory.",
    )
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--temporal-annotation-file", type=Path, default=None)
    parser.add_argument("--normal-label", type=int, default=None)
    parser.add_argument("--missing-score", type=float, default=0.0)
    parser.add_argument(
        "--segments-only",
        action="store_true",
        help="Evaluate only frames covered by scored segments. Useful for smoke tests with --segment-limit.",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Ignore per-video JSON outputs that are not marked complete.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    annotation_file = args.annotation_file or defaults["annotation_file"]
    temporal_annotation_file = args.temporal_annotation_file or defaults["temporal_annotation_file"]
    normal_label = args.normal_label if args.normal_label is not None else int(defaults["normal_label"])

    videos = load_agentic_results(args.results_json, complete_only=args.complete_only)
    records = load_video_records(annotation_file)
    temporal_annotations = load_temporal_annotations(temporal_annotation_file)

    flat_scores: list[float] = []
    flat_binary_labels: list[int] = []
    per_video: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for result in videos:
        name = normalize_name(result.get("video_name", ""))
        if name not in records:
            skipped[name] = "missing in anomaly_test annotation file"
            continue
        if name not in temporal_annotations:
            skipped[name] = "missing in temporal annotation file"
            continue

        record = records[name]
        labels = make_frame_labels(record, temporal_annotations[name], normal_label)
        scores, covered = expand_segment_scores(
            result,
            num_frames=len(labels),
            missing_score=args.missing_score,
        )
        if args.segments_only:
            scores = [score for score, keep in zip(scores, covered) if keep]
            labels = [label for label, keep in zip(labels, covered) if keep]

        if not scores:
            skipped[name] = "no scored frames"
            continue

        binary_labels = [int(label != normal_label) for label in labels]
        metrics = compute_metrics(scores, binary_labels)
        per_video[name] = {
            "num_frames": len(scores),
            "positive_frames": int(sum(binary_labels)),
            "negative_frames": int(len(binary_labels) - sum(binary_labels)),
            "metrics": metrics,
        }
        flat_scores.extend(scores)
        flat_binary_labels.extend(binary_labels)

    dataset_metrics = compute_metrics(flat_scores, flat_binary_labels)
    payload = {
        "dataset": args.dataset,
        "normal_label": normal_label,
        "results_json": str(args.results_json),
        "annotation_file": str(annotation_file),
        "temporal_annotation_file": str(temporal_annotation_file),
        "segments_only": args.segments_only,
        "num_videos": len(per_video),
        "num_frames": len(flat_scores),
        "positive_frames": int(sum(flat_binary_labels)),
        "negative_frames": int(len(flat_binary_labels) - sum(flat_binary_labels)),
        "dataset_metrics": dataset_metrics,
        "per_video": per_video,
        "skipped": skipped,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["dataset_metrics"], indent=2))
    print(
        f"videos={payload['num_videos']} frames={payload['num_frames']} "
        f"positive={payload['positive_frames']} negative={payload['negative_frames']}"
    )
    if skipped:
        print(f"skipped={len(skipped)}")


def load_agentic_results(path: Path, complete_only: bool) -> list[dict[str, Any]]:
    if path.is_dir():
        aggregate = path / "agentic_results.json"
        if aggregate.exists():
            videos = _videos_from_payload(read_json(aggregate), complete_only)
            if videos:
                return videos
        videos = []
        for item in sorted(path.glob("*.json")):
            if item.name == "agentic_results.json" or item.name.startswith("metrics"):
                continue
            payload = read_json(item)
            if "segments" in payload and (not complete_only or payload.get("complete") is True):
                videos.append(payload)
        return videos

    payload = read_json(path)
    if "vid_score" in payload:
        return _videos_from_vid_score(payload)
    if "videos" in payload:
        return _videos_from_payload(payload, complete_only)
    if "segments" in payload:
        if complete_only and payload.get("complete") is not True:
            return []
        return [payload]
    raise ValueError(f"Unrecognized VAD-Agent-MIF result format: {path}")


def _videos_from_payload(payload: dict[str, Any], complete_only: bool) -> list[dict[str, Any]]:
    videos = []
    for key, value in payload.get("videos", {}).items():
        if not isinstance(value, dict):
            continue
        if complete_only and value.get("complete") is not True:
            continue
        if "video_name" not in value:
            value = {**value, "video_name": key}
        videos.append(value)
    return videos


def _videos_from_vid_score(payload: dict[str, Any]) -> list[dict[str, Any]]:
    videos = []
    for video_name, segment_scores in payload.get("vid_score", {}).items():
        if not isinstance(segment_scores, dict):
            continue
        segments = []
        for edge, value in segment_scores.items():
            parsed_edge = parse_edge(edge)
            if parsed_edge is None:
                continue
            score = value[0] if isinstance(value, list) and value else value
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None
            if score is not None and score < 0:
                score = None
            start, end = parsed_edge
            segments.append(
                {
                    "plan": {"start": start, "end": end},
                    "anomaly_score": score,
                }
            )
        videos.append({"video_name": video_name, "segments": segments, "complete": True})
    return videos


def parse_edge(edge: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", str(edge))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_video_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        records[normalize_name(parts[0])] = {
            "start_frame": int(parts[1]),
            "end_frame": int(parts[2]),
            "labels": [int(label) for label in parts[3].split(",")],
        }
    return records


def load_temporal_annotations(path: Path) -> dict[str, list[str]]:
    annotations = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        annotations[normalize_name(parts[0])] = [value for value in parts[2:] if value != "-1"]
    return annotations


def make_frame_labels(record: dict[str, Any], annotations: list[str], normal_label: int) -> list[int]:
    num_frames = record["end_frame"] - record["start_frame"] + 1
    labels = [normal_label] * num_frames
    starts = annotations[::2]
    ends = annotations[1::2]
    event_labels = list(record["labels"])
    if not starts:
        return labels
    if len(event_labels) < len(starts):
        event_labels.extend([event_labels[-1]] * (len(starts) - len(event_labels)))
    for start_text, end_text, event_label in zip(starts, ends, event_labels):
        start = max(0, int(start_text) - record["start_frame"])
        end = min(num_frames - 1, int(end_text) - record["start_frame"])
        if end < start:
            continue
        for idx in range(start, end + 1):
            labels[idx] = event_label
    return labels


def expand_segment_scores(
    result: dict[str, Any],
    num_frames: int,
    missing_score: float,
) -> tuple[list[float], list[bool]]:
    scores = [float(missing_score)] * num_frames
    covered = [False] * num_frames
    for segment in result.get("segments", []):
        score = segment.get("anomaly_score")
        if score is None:
            score, _ = parse_reasoning_json(str(segment.get("reasoning_response") or ""))
            if score is None:
                continue
        plan = segment.get("plan", {})
        start = max(0, int(plan.get("start", 0)))
        end = min(num_frames - 1, int(plan.get("end", start)))
        if end < start:
            continue
        score_value = max(0.0, min(1.0, float(score)))
        for idx in range(start, end + 1):
            scores[idx] = score_value
            covered[idx] = True
    return scores, covered


def compute_metrics(scores: list[float], binary_labels: list[int]) -> dict[str, Any]:
    if not scores:
        return {"roc_auc": None, "average_precision": None, "pr_auc": None}
    y_score = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(binary_labels, dtype=np.int32)
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
            "pr_auc": None,
            "reason": "needs both normal and abnormal frames",
        }
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 6),
        "pr_auc": round(float(auc(recall, precision)), 6),
    }


def normalize_name(name: str) -> str:
    return Path(str(name)).stem


if __name__ == "__main__":
    main()

