from __future__ import annotations

import argparse
import copy
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score


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
    parser = argparse.ArgumentParser(
        description="Apply VADTree-style similarity refinement to a vid_score JSON and optionally evaluate it."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--scores-json", type=Path, required=True, help="Input JSON with vid_score.")
    parser.add_argument("--similarity-pkl", type=Path, required=True, help="VADTree/ImageBind sim_*.pkl.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional refined vid_score JSON to write.")
    parser.add_argument("--similarity-type", type=str, default="VxV")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--dyn-ratio", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--score-index",
        type=int,
        default=0,
        help="List index to refine. DeepSeek Agentic outputs use 0. VADTree refined files use 1 for raw LLM.",
    )
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--temporal-annotation-file", type=Path, default=None)
    parser.add_argument("--normal-label", type=int, default=None)
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_json(args.scores_json)
    refined_payload, stats = refine_payload(
        payload=payload,
        similarity_pkl=args.similarity_pkl,
        similarity_type=args.similarity_type,
        top_k=args.top_k,
        num_neighbors=args.num_neighbors,
        dyn_ratio=args.dyn_ratio,
        tau=args.tau,
        score_index=args.score_index,
    )

    refined_payload["refinement_config"] = {
        "scores_json": str(args.scores_json),
        "similarity_pkl": str(args.similarity_pkl),
        "similarity_type": args.similarity_type,
        "top_k": args.top_k,
        "num_neighbors": args.num_neighbors,
        "dyn_ratio": args.dyn_ratio,
        "tau": args.tau,
        "score_index": args.score_index,
        "stats": stats,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(refined_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved={args.output_json.resolve()}")

    print(f"refined_videos={stats['refined_videos']} skipped_videos={stats['skipped_videos']}")
    if stats["skipped"]:
        print(f"first_skipped={stats['skipped'][:5]}")

    if not args.no_eval:
        defaults = DATASET_DEFAULTS[args.dataset]
        annotation_file = args.annotation_file or defaults["annotation_file"]
        temporal_annotation_file = args.temporal_annotation_file or defaults["temporal_annotation_file"]
        normal_label = args.normal_label if args.normal_label is not None else int(defaults["normal_label"])
        metrics = evaluate_vid_score(refined_payload, annotation_file, temporal_annotation_file, normal_label)
        print(json.dumps(metrics, indent=2))


def refine_payload(
    payload: dict[str, Any],
    similarity_pkl: Path,
    similarity_type: str,
    top_k: int,
    num_neighbors: int,
    dyn_ratio: float | None,
    tau: float,
    score_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with similarity_pkl.open("rb") as f:
        sim_payload = pickle.load(f)
    vid_sim = sim_payload["vid_sim"]

    refined_payload = copy.deepcopy(payload)
    refined_payload.setdefault("vid_score", {})
    stats: dict[str, Any] = {"refined_videos": 0, "skipped_videos": 0, "skipped": []}

    for video_key, segment_scores in payload.get("vid_score", {}).items():
        if video_key not in vid_sim:
            stats["skipped_videos"] += 1
            stats["skipped"].append({"video": video_key, "reason": "missing similarity"})
            continue
        if similarity_type not in vid_sim[video_key]:
            stats["skipped_videos"] += 1
            stats["skipped"].append({"video": video_key, "reason": f"missing {similarity_type}"})
            continue

        raw_scores = []
        segment_keys = list(segment_scores.keys())
        for key in segment_keys:
            raw_scores.append(extract_score(segment_scores[key], score_index))

        sim_matrix = np.asarray(vid_sim[video_key][similarity_type])
        if sim_matrix.shape[0] != len(raw_scores):
            stats["skipped_videos"] += 1
            stats["skipped"].append(
                {"video": video_key, "reason": f"shape mismatch scores={len(raw_scores)} sim={list(sim_matrix.shape)}"}
            )
            continue

        refined_scores = calculate_refined_scores(
            raw_scores=np.asarray(raw_scores, dtype=np.float32),
            sim_matrix=sim_matrix,
            top_k=top_k,
            num_neighbors=num_neighbors,
            dyn_ratio=dyn_ratio,
            tau=tau,
        )

        out_video_scores = {}
        for key, refined_score in zip(segment_keys, refined_scores):
            old_value = segment_scores[key]
            if isinstance(old_value, list):
                out_video_scores[key] = [round(float(refined_score), 4), *old_value]
            else:
                out_video_scores[key] = [round(float(refined_score), 4), old_value]
        refined_payload["vid_score"][video_key] = out_video_scores
        stats["refined_videos"] += 1

    return refined_payload, stats


def calculate_refined_scores(
    raw_scores: np.ndarray,
    sim_matrix: np.ndarray,
    top_k: int,
    num_neighbors: int,
    dyn_ratio: float | None,
    tau: float,
) -> list[float]:
    if dyn_ratio is not None:
        num_neighbors = max(int(raw_scores.shape[0] * dyn_ratio), num_neighbors)
    refined = []
    for idx in range(raw_scores.shape[0]):
        left = max(0, idx - num_neighbors // 2)
        right = idx + num_neighbors // 2
        local_sims = sim_matrix[idx][left:right]
        local_scores = raw_scores[left:right]
        if len(local_scores) == 0:
            refined.append(float(raw_scores[idx]))
            continue
        order = local_sims.argsort()[::-1][: min(top_k, len(local_sims))]
        top_sims = local_sims[order]
        top_scores = local_scores[order]
        stable = top_sims - np.max(top_sims)
        weights = np.exp(stable / tau)
        weights = weights / np.sum(weights)
        refined.append(float(np.sum(top_scores * weights)))
    return refined


def evaluate_vid_score(
    payload: dict[str, Any],
    annotation_file: Path,
    temporal_annotation_file: Path,
    normal_label: int,
) -> dict[str, Any]:
    records = load_video_records(annotation_file)
    temporal = load_temporal_annotations(temporal_annotation_file)
    flat_scores: list[float] = []
    flat_labels: list[int] = []
    skipped = {}

    for video_key, segment_scores in payload.get("vid_score", {}).items():
        name = Path(video_key).stem
        if name not in records or name not in temporal:
            skipped[name] = "missing annotations"
            continue
        labels = make_frame_labels(records[name], temporal[name], normal_label)
        scores = [0.0] * len(labels)
        for edge, value in segment_scores.items():
            parsed = parse_edge(edge)
            if parsed is None:
                continue
            start, end = parsed
            start = max(0, start)
            end = min(len(labels) - 1, end)
            score = extract_score(value, 0)
            if score < 0 or end < start:
                continue
            scores[start : end + 1] = [float(score)] * (end - start + 1)
        flat_scores.extend(scores)
        flat_labels.extend(int(label != normal_label) for label in labels)

    metrics = compute_metrics(flat_scores, flat_labels)
    metrics.update(
        {
            "videos": len(payload.get("vid_score", {})) - len(skipped),
            "frames": len(flat_scores),
            "positive_frames": int(sum(flat_labels)),
            "negative_frames": int(len(flat_labels) - sum(flat_labels)),
            "skipped": skipped,
        }
    )
    return metrics


def extract_score(value: Any, score_index: int) -> float:
    if isinstance(value, list):
        if score_index >= len(value):
            return -1.0
        value = value[score_index]
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def compute_metrics(scores: list[float], labels: list[int]) -> dict[str, float | None]:
    if not scores or len(set(labels)) < 2:
        return {"roc_auc": None, "average_precision": None, "pr_auc": None}
    y_score = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.int32)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 6),
        "pr_auc": round(float(auc(recall, precision)), 6),
    }


def load_video_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        records[Path(parts[0]).stem] = {
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
        annotations[Path(parts[0]).stem] = [value for value in parts[2:] if value != "-1"]
    return annotations


def make_frame_labels(record: dict[str, Any], annotations: list[str], normal_label: int) -> list[int]:
    num_frames = record["end_frame"] - record["start_frame"] + 1
    labels = [normal_label] * num_frames
    starts = annotations[::2]
    ends = annotations[1::2]
    event_labels = list(record["labels"])
    if starts and len(event_labels) < len(starts):
        event_labels.extend([event_labels[-1]] * (len(starts) - len(event_labels)))
    for start_text, end_text, event_label in zip(starts, ends, event_labels):
        start = max(0, int(start_text) - record["start_frame"])
        end = min(num_frames - 1, int(end_text) - record["start_frame"])
        if end >= start:
            labels[start : end + 1] = [event_label] * (end - start + 1)
    return labels


def parse_edge(edge: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", str(edge))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

