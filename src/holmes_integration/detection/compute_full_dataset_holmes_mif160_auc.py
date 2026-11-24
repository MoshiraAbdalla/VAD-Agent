#!/usr/bin/env python3
"""Full UCF/XD Holmes-CLS-MIF + ATS AUC/AP evaluation.

This extends the HIVAU-only Holmes-CLS-MIF experiment to the full VAD-Agent
test lists:

* UCF: data/UCF_Eval.json
* XD:  data/xd_test_800.json + data/name_gt_pair.json

The selected-frame cache format intentionally matches
compute_hivau_holmescls_mif160_ats_direct_auc.py so already computed HIVAU
subset caches can be copied and reused.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_hivau_holmes_mif160_ats_direct_auc import (  # noqa: E402
    build_projection_ranges,
    load_json,
    metric_pair,
    project_segments,
    save_json,
    select_and_score_video,
)
from holmesvau.holmesvau_utils import load_model  # noqa: E402


def safe_name(text):
    return "".join("_" if ch in '<>:"/\\|?*' else ch for ch in str(text))


def strip_video_extension(text):
    value = str(text).replace("\\", "/").rsplit("/", 1)[-1]
    lower = value.lower()
    for suffix in (".mp4", ".avi", ".mov", ".mkv"):
        if lower.endswith(suffix):
            return value[: -len(suffix)]
    return value


def cache_name(dataset, key):
    return f"{dataset}_{safe_name(key)}.json"


def build_ucf_video_index(ucf_root):
    index = {}
    for dirpath, _, filenames in os.walk(ucf_root):
        for name in filenames:
            if name.lower().endswith((".mp4", ".avi")):
                stem = Path(name).stem
                index.setdefault(stem, os.path.join(dirpath, name))
    return index


def resolve_ucf_video(sample, ucf_root, ucf_index):
    video = sample["video"]
    key = strip_video_extension(video)
    direct = Path(ucf_root) / video
    if direct.exists():
        return key, str(direct)
    indexed = ucf_index.get(key)
    if indexed:
        return key, indexed
    return key, ""


def resolve_xd_video(sample, xd_root):
    key = strip_video_extension(sample["video"])
    direct = Path(xd_root) / "videos" / f"{key}.mp4"
    if direct.exists():
        return key, str(direct)
    direct = Path(xd_root) / f"{key}.mp4"
    if direct.exists():
        return key, str(direct)
    return key, ""


def gt_from_temporal_label(sample):
    length = int(sample["length"])
    gt = np.zeros((length,), dtype=np.uint8)
    labels = sample.get("temporal_label", [])
    for i in range(0, len(labels), 2):
        if i + 1 >= len(labels) or int(labels[i]) == -1:
            continue
        start = max(0, int(labels[i]))
        end = min(length, int(labels[i + 1]))
        if end > start:
            gt[start:end] = 1
    return gt


def load_ucf_items(ann_path, ucf_root):
    rows = load_json(ann_path)
    ucf_index = build_ucf_video_index(ucf_root)
    items = []
    for row in rows:
        key, video_path = resolve_ucf_video(row, ucf_root, ucf_index)
        items.append(
            {
                "dataset": "ucf",
                "key": key,
                "annotation_video": row["video"],
                "video_path": video_path,
                "gt": gt_from_temporal_label(row),
                "temporal_label": row.get("temporal_label"),
            }
        )
    return items


def load_xd_items(ann_path, xd_root, name_gt_pair_path=None):
    rows = load_json(ann_path)
    gt_pair = load_json(name_gt_pair_path) if name_gt_pair_path else {}
    items = []
    for row in rows:
        key, video_path = resolve_xd_video(row, xd_root)
        if key in gt_pair:
            gt = np.asarray(gt_pair[key], dtype=np.uint8)
        else:
            gt = gt_from_temporal_label(row)
        items.append(
            {
                "dataset": "xd",
                "key": key,
                "annotation_video": row["video"],
                "video_path": video_path,
                "gt": gt,
                "temporal_label": row.get("temporal_label"),
                "label": row.get("label"),
            }
        )
    return items


def load_items(args):
    if args.dataset == "ucf":
        return load_ucf_items(args.ann_root, args.ucf_root)
    if args.dataset == "xd":
        return load_xd_items(args.ann_root, args.xd_root, args.name_gt_pair)
    raise ValueError(args.dataset)


def reuse_existing_caches(items, dataset, frame_scores_dir, reuse_cache_dirs):
    frame_scores_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "requested_items": len(items),
        "already_present": 0,
        "copied": 0,
        "not_found": 0,
        "sources": [str(Path(p)) for p in reuse_cache_dirs],
        "copied_items": [],
    }
    source_dirs = [Path(p) for p in reuse_cache_dirs if p]
    for item in items:
        dst = frame_scores_dir / cache_name(dataset, item["key"])
        if dst.exists():
            report["already_present"] += 1
            continue
        copied = False
        for source_dir in source_dirs:
            src = source_dir / cache_name(dataset, item["key"])
            if src.exists():
                shutil.copy2(src, dst)
                report["copied"] += 1
                report["copied_items"].append(item["key"])
                copied = True
                break
        if not copied:
            report["not_found"] += 1
    return report


def evaluate_cached_or_run(item, args, model, sampler, frame_scores_dir):
    cache_path = frame_scores_dir / cache_name(args.dataset, item["key"])
    if args.resume and cache_path.exists():
        cache = load_json(cache_path)
        selected_indices = (cache.get("selected_mif_indices") or cache.get("selected_maf_indices") or [])
        selected_scores = (cache.get("selected_mif_scores") or cache.get("selected_maf_scores"))
        if selected_scores is not None and "segments" in cache:
            score_by_frame = {
                int(frame_idx): float(score)
                for frame_idx, score in zip(selected_indices, selected_scores)
            }
            frame_scores, covered_frames = project_segments(
                int(cache.get("length", len(cache.get("frame_scores", [])))),
                cache["segments"],
                score_by_frame,
                args.projection,
            )
        else:
            frame_scores = np.asarray(cache["frame_scores"], dtype=np.float32)
            covered_frames = int(cache.get("covered_frames", int(np.count_nonzero(frame_scores))))
        return frame_scores, selected_indices, covered_frames, cache

    length, segments, selected_indices, selected_scores = select_and_score_video(
        video_path=item["video_path"],
        model=model,
        sampler=sampler,
        segment_len=args.segment_len,
        segment_stride=args.segment_stride,
        dense_equivalent_stride=args.dense_equivalent_stride,
        mif_w=args.mif_w,
        batch_size=args.batch_size,
    )
    score_by_frame = {int(frame_idx): float(score) for frame_idx, score in zip(selected_indices, selected_scores)}
    frame_scores, covered_frames = project_segments(length, segments, score_by_frame, args.projection)
    cache = {
        "dataset": args.dataset,
        "part": args.dataset,
        "video": item["key"],
        "annotation_video": item["annotation_video"],
        "video_path": item["video_path"],
        "length": int(length),
        "feature_space": "HolmesVAU model.vision_model CLS token",
        "segment_len": args.segment_len,
        "segment_stride": args.segment_stride,
        "dense_equivalent_stride": args.dense_equivalent_stride,
        "mif_w": args.mif_w,
        "projection": args.projection,
        "segments": segments,
        "selected_mif_indices": selected_indices,
        "selected_mif_scores": np.asarray(selected_scores, dtype=np.float32).tolist(),
        "projection_ranges": build_projection_ranges(length, segments, score_by_frame, args.projection),
        "covered_frames": int(covered_frames),
        "frame_scores": frame_scores.tolist(),
    }
    save_json(cache_path, cache)
    return frame_scores, selected_indices, covered_frames, cache


def align_scores_to_gt_length(frame_scores, gt_length):
    """Return one score for every GT frame, with explicit alignment metadata."""
    scores = np.asarray(frame_scores, dtype=np.float32)
    gt_length = int(gt_length)
    if gt_length <= 0:
        raise ValueError(f"invalid gt_length={gt_length}")
    if scores.size == 0:
        raise ValueError("empty frame_scores")
    if scores.size == gt_length:
        return scores, {
            "method": "identity",
            "raw_score_length": int(scores.size),
            "score_length": int(scores.size),
            "gt_length": gt_length,
        }

    if scores.size == 1:
        aligned = np.full((gt_length,), float(scores[0]), dtype=np.float32)
    else:
        source_x = np.linspace(0.0, 1.0, num=scores.size, dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, num=gt_length, dtype=np.float64)
        aligned = np.interp(target_x, source_x, scores.astype(np.float64)).astype(np.float32)

    return aligned, {
        "method": "linear_resample_to_gt_length",
        "raw_score_length": int(scores.size),
        "score_length": int(aligned.size),
        "gt_length": gt_length,
    }


def main():
    parser = argparse.ArgumentParser(description="Full UCF/XD Holmes-CLS-MIF + ATS direct AUC/AP.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--ckpt", default="HolmesVAU-2B")
    parser.add_argument("--sampler_path", default=os.path.join("holmesvau", "ATS", "anomaly_scorer.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ann_root", required=True)
    parser.add_argument("--ucf_root", default="data/ucf/videos")
    parser.add_argument("--xd_root", default="data/xd/videos")
    parser.add_argument("--name_gt_pair", default="data/annotations/name_gt_pair.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame_scores_dir", required=True)
    parser.add_argument("--reuse_cache_dir", action="append", default=[])
    parser.add_argument("--segment_len", type=int, default=160)
    parser.add_argument("--segment_stride", type=int, default=160)
    parser.add_argument("--dense_equivalent_stride", type=int, default=16)
    parser.add_argument("--mif_w", type=int, default=8)
    parser.add_argument("--projection", choices=["nearest", "linear", "segment_mean", "expmax64", "expmax128"], default="nearest")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copy_only", action="store_true")
    args = parser.parse_args()

    items = load_items(args)
    if args.max_videos and args.max_videos > 0:
        items = items[: args.max_videos]

    frame_scores_dir = Path(args.frame_scores_dir)
    reuse_report = reuse_existing_caches(items, args.dataset, frame_scores_dir, args.reuse_cache_dir)
    save_json(Path(args.out).with_suffix(".cache_reuse.json"), reuse_report)
    print(
        f"Loaded {len(items)} {args.dataset} items; cache already_present={reuse_report['already_present']} "
        f"copied={reuse_report['copied']} missing={reuse_report['not_found']}",
        flush=True,
    )
    if args.copy_only:
        return

    all_caches_available = args.resume and all(
        (frame_scores_dir / cache_name(args.dataset, item["key"])).exists()
        for item in items
    )
    if all_caches_available:
        print("All caches available; skipping HolmesVAU + ATS load.", flush=True)
        model = tokenizer = generation_config = sampler = None
    else:
        print("Loading HolmesVAU + ATS...", flush=True)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, tokenizer, generation_config, sampler = load_model(args.ckpt, args.sampler_path, device)
        sampler.anomaly_scorer.eval()

    all_gt = []
    all_scores = []
    per_video = []
    missing = []
    for idx, item in enumerate(items, start=1):
        if not item["video_path"] or not os.path.exists(item["video_path"]):
            missing.append({"video": item["key"], "reason": f"video path not found: {item['video_path']}"})
            continue
        cache_path = frame_scores_dir / cache_name(args.dataset, item["key"])
        if args.resume and cache_path.exists():
            print(f"[{idx}/{len(items)}] {args.dataset}/{item['key']}: using cache", flush=True)
        else:
            print(f"[{idx}/{len(items)}] {args.dataset}/{item['key']}: Holmes-CLS MIF-160 selection", flush=True)
        try:
            frame_scores, selected_indices, covered_frames, cache = evaluate_cached_or_run(
                item, args, model, sampler, frame_scores_dir
            )
        except Exception as exc:
            missing.append({"video": item["key"], "reason": str(exc)})
            print(f"[{idx}/{len(items)}] {args.dataset}/{item['key']}: failed: {exc}", flush=True)
            continue

        gt_eval = np.asarray(item["gt"], dtype=np.uint8)
        score_eval, alignment = align_scores_to_gt_length(frame_scores, len(gt_eval))
        if len(score_eval) != len(gt_eval):
            raise RuntimeError(
                f"score/GT length mismatch after alignment for {item['key']}: "
                f"scores={len(score_eval)} gt={len(gt_eval)}"
            )
        metrics = metric_pair(gt_eval, score_eval)
        print(
            f"[{idx}/{len(items)}] {args.dataset}/{item['key']}: selected_mif_frames={len(selected_indices)} "
            f"score_length={len(score_eval)} gt_length={len(gt_eval)} "
            f"AUC={metrics.get('AUC')} AP={metrics.get('AP')}",
            flush=True,
        )
        per_video.append(
            {
                "dataset": args.dataset,
                "video": item["key"],
                "annotation_video": item["annotation_video"],
                "video_path": item["video_path"],
                "length": int(len(gt_eval)),
                "gt_length": int(len(gt_eval)),
                "raw_score_length": alignment["raw_score_length"],
                "score_length": alignment["score_length"],
                "score_alignment": alignment["method"],
                "positive_frames": int(gt_eval.sum()),
                "selected_mif_frames": len(selected_indices),
                "covered_frames": int(covered_frames),
                "segment_len": args.segment_len,
                "segment_stride": args.segment_stride,
                "projection": args.projection,
                **metrics,
            }
        )
        all_gt.append(gt_eval)
        all_scores.append(score_eval)

    if not all_gt:
        raise SystemExit("No videos were evaluated.")

    all_gt_arr = np.concatenate(all_gt)
    all_score_arr = np.concatenate(all_scores)
    valid_auc = [item["AUC"] for item in per_video if item.get("AUC") is not None]
    valid_ap = [item["AP"] for item in per_video if item.get("AP") is not None]
    output = {
        "dataset": args.dataset,
        "ann_root": os.path.normpath(args.ann_root),
        "frame_scores_dir": os.path.normpath(args.frame_scores_dir),
        "feature_space": "HolmesVAU model.vision_model CLS token",
        "segment_len": args.segment_len,
        "segment_stride": args.segment_stride,
        "dense_equivalent_stride": args.dense_equivalent_stride,
        "mif_w": args.mif_w,
        "projection": args.projection,
        "cache_reuse": reuse_report,
        "num_videos_requested": len(items),
        "num_videos_evaluated": len(per_video),
        "num_videos_missing": len(missing),
        "total_frames": int(all_gt_arr.size),
        "positive_frames": int(all_gt_arr.sum()),
        "overall": metric_pair(all_gt_arr, all_score_arr),
        "macro_per_video": {
            "AUC": None if not valid_auc else float(np.mean(valid_auc)),
            "AP": None if not valid_ap else float(np.mean(valid_ap)),
            "num_videos": len(valid_auc),
        },
        "per_video": per_video,
        "missing": missing,
    }
    save_json(args.out, output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()




