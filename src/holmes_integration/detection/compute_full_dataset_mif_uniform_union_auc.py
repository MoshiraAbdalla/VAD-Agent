#!/usr/bin/env python3
"""Full UCF/XD MIF + uniform union frame scoring with Holmes anomaly scorer.

This experiment uses saved MIF-selected frame indices, unions them with uniform
stride frames, reruns the Holmes anomaly scorer on that combined sequence, then
projects those union-sequence scores back to raw frames for AUC/AP.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_full_dataset_holmes_mif160_auc import (  # noqa: E402
    align_scores_to_gt_length,
    cache_name,
    load_items,
    load_json,
    metric_pair,
    save_json,
)
from compute_hivau_holmes_mif160_ats_direct_auc import encode_holmes_cls  # noqa: E402
from holmesvau.holmesvau_utils import load_model  # noqa: E402


def union_frame_indices(n_frames, mif_indices, uniform_stride):
    uniform_indices = list(range(0, int(n_frames), int(uniform_stride)))
    mif_indices = [int(idx) for idx in mif_indices if 0 <= int(idx) < int(n_frames)]
    return sorted(set(uniform_indices).union(mif_indices)), uniform_indices, mif_indices


def align_sequence_scores(scores, target_len):
    scores = np.asarray(scores, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"invalid target_len={target_len}")
    if scores.size == target_len:
        return scores, {
            "method": "identity",
            "raw_score_length": int(scores.size),
            "score_length": int(scores.size),
            "target_length": target_len,
        }
    if scores.size == 0:
        raise ValueError("empty anomaly score sequence")
    if scores.size == 1:
        aligned = np.full((target_len,), float(scores[0]), dtype=np.float32)
    else:
        source_x = np.linspace(0.0, 1.0, num=scores.size, dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
        aligned = np.interp(target_x, source_x, scores.astype(np.float64)).astype(np.float32)
    return aligned, {
        "method": "linear_resample_to_union_length",
        "raw_score_length": int(scores.size),
        "score_length": int(aligned.size),
        "target_length": target_len,
    }


def project_points(length, point_indices, point_scores, mode):
    length = int(length)
    point_indices = np.asarray(point_indices, dtype=np.int32)
    point_scores = np.asarray(point_scores, dtype=np.float32)
    valid = (point_indices >= 0) & (point_indices < length)
    point_indices = point_indices[valid]
    point_scores = point_scores[valid]
    if length <= 0:
        return np.asarray([], dtype=np.float32)
    if point_scores.size == 0:
        return np.zeros((length,), dtype=np.float32)
    order = np.argsort(point_indices, kind="mergesort")
    point_indices = point_indices[order]
    point_scores = point_scores[order]
    frames = np.arange(length, dtype=np.float32)

    if mode == "linear":
        return np.interp(
            frames,
            point_indices.astype(np.float32),
            point_scores,
            left=float(point_scores[0]),
            right=float(point_scores[-1]),
        ).astype(np.float32)

    if mode == "nearest":
        right = np.searchsorted(point_indices, np.arange(length, dtype=np.int32), side="left")
        right = np.clip(right, 0, len(point_indices) - 1)
        left = np.clip(right - 1, 0, len(point_indices) - 1)
        frame_ids = np.arange(length, dtype=np.int32)
        use_right = np.abs(point_indices[right] - frame_ids) < np.abs(frame_ids - point_indices[left])
        nearest = np.where(use_right, right, left)
        return point_scores[nearest].astype(np.float32)

    if mode.startswith("expmax"):
        tau = float(mode.replace("expmax", ""))
        distances = np.abs(frames[:, None] - point_indices.astype(np.float32)[None, :])
        return np.max(point_scores[None, :] * np.exp(-distances / tau), axis=1).astype(np.float32)

    raise ValueError(f"unknown projection mode: {mode}")


@torch.inference_mode()
def score_union_sequence(video_path, union_indices, model, sampler, batch_size):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n_frames = len(vr)
    union_indices = [int(idx) for idx in union_indices if 0 <= int(idx) < n_frames]
    if not union_indices:
        return n_frames, [], np.asarray([], dtype=np.float32)
    cls_tokens = encode_holmes_cls(vr, union_indices, model, batch_size)
    vid_feats = cls_tokens.unsqueeze(0).to(sampler.device)
    scores = sampler.anomaly_scorer(vid_feats)["anomaly_scores"][0].detach().cpu().numpy().astype(np.float32)
    return n_frames, union_indices, scores


def evaluate_cached_or_run(item, args, model, sampler, union_cache_dir):
    cache_path = union_cache_dir / cache_name(args.dataset, item["key"])
    if args.resume and cache_path.exists():
        return load_json(cache_path)

    mif_cache_path = Path(args.mif_frame_scores_dir) / cache_name(args.dataset, item["key"])
    if not mif_cache_path.exists():
        raise FileNotFoundError(f"MIF cache not found: {mif_cache_path}")
    mif_cache = load_json(mif_cache_path)
    mif_length = int(mif_cache.get("length", len(mif_cache.get("frame_scores", []))))
    planned_union_indices, uniform_indices, mif_indices = union_frame_indices(
        mif_length,
        (mif_cache.get("selected_mif_indices") or mif_cache.get("selected_maf_indices") or []),
        args.uniform_stride,
    )
    n_frames, union_indices, raw_scores = score_union_sequence(
        item["video_path"],
        planned_union_indices,
        model,
        sampler,
        args.batch_size,
    )
    union_scores, sequence_alignment = align_sequence_scores(raw_scores, len(union_indices))
    cache = {
        "dataset": args.dataset,
        "video": item["key"],
        "annotation_video": item["annotation_video"],
        "video_path": item["video_path"],
        "n_frames": int(n_frames),
        "uniform_stride": int(args.uniform_stride),
        "uniform_indices": [int(idx) for idx in uniform_indices if 0 <= int(idx) < int(n_frames)],
        "mif_indices": [int(idx) for idx in mif_indices if 0 <= int(idx) < int(n_frames)],
        "union_indices": [int(idx) for idx in union_indices],
        "raw_anomaly_scores": np.asarray(raw_scores, dtype=np.float32).tolist(),
        "union_scores": np.asarray(union_scores, dtype=np.float32).tolist(),
        "sequence_alignment": sequence_alignment,
        "feature_space": "HolmesVAU model.vision_model CLS token",
        "scorer": "HolmesVAU ATS/URDMU anomaly_scorer.pth",
        "selection": "saved MIF selected frames union uniform stride frames",
    }
    save_json(cache_path, cache)
    return cache


def main():
    parser = argparse.ArgumentParser(description="Full-dataset MIF+uniform union frame scoring AUC/AP.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--ckpt", default="HolmesVAU-2B")
    parser.add_argument("--sampler_path", default=os.path.join("holmesvau", "ATS", "anomaly_scorer.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ann_root", required=True)
    parser.add_argument("--ucf_root", default="data/ucf/videos")
    parser.add_argument("--xd_root", default="data/xd/videos")
    parser.add_argument("--name_gt_pair", default="data/annotations/name_gt_pair.json")
    parser.add_argument("--mif_frame_scores_dir", required=True)
    parser.add_argument("--union_cache_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--uniform_stride", type=int, default=16)
    parser.add_argument("--projection", choices=["linear", "nearest", "expmax64", "expmax128"], default="linear")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    items = load_items(args)
    if args.max_videos and args.max_videos > 0:
        items = items[: args.max_videos]

    union_cache_dir = Path(args.union_cache_dir)
    union_cache_dir.mkdir(parents=True, exist_ok=True)

    all_caches_available = args.resume and all(
        (union_cache_dir / cache_name(args.dataset, item["key"])).exists()
        for item in items
    )
    if all_caches_available:
        print("All union caches available; skipping HolmesVAU + anomaly scorer load.", flush=True)
        model = sampler = None
    else:
        print("Loading HolmesVAU + anomaly scorer...", flush=True)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, _tokenizer, _generation_config, sampler = load_model(args.ckpt, args.sampler_path, device)
        sampler.anomaly_scorer.eval()

    all_gt = []
    all_scores = []
    per_video = []
    missing = []

    for idx, item in enumerate(items, start=1):
        cache_path = union_cache_dir / cache_name(args.dataset, item["key"])
        using_cache = args.resume and cache_path.exists()
        if not using_cache and (not item["video_path"] or not os.path.exists(item["video_path"])):
            missing.append({"video": item["key"], "reason": f"video path not found: {item['video_path']}"})
            continue
        mode = "using union cache" if args.resume and cache_path.exists() else "scoring union frames"
        print(f"[{idx}/{len(items)}] {args.dataset}/{item['key']}: {mode}", flush=True)
        try:
            cache = evaluate_cached_or_run(item, args, model, sampler, union_cache_dir)
        except Exception as exc:
            missing.append({"video": item["key"], "reason": str(exc)})
            print(f"[{idx}/{len(items)}] {args.dataset}/{item['key']}: failed: {exc}", flush=True)
            continue

        gt_eval = np.asarray(item["gt"], dtype=np.uint8)
        projected = project_points(
            len(gt_eval),
            cache["union_indices"],
            cache["union_scores"],
            args.projection,
        )
        score_eval, score_alignment = align_scores_to_gt_length(projected, len(gt_eval))
        metrics = metric_pair(gt_eval, score_eval)
        print(
            f"[{idx}/{len(items)}] {args.dataset}/{item['key']}: "
            f"uniform={len(cache.get('uniform_indices', []))} mif={len(cache.get('mif_indices', []))} "
            f"union={len(cache.get('union_indices', []))} AUC={metrics.get('AUC')} AP={metrics.get('AP')}",
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
                "positive_frames": int(gt_eval.sum()),
                "uniform_frames": len(cache.get("uniform_indices", [])),
                "mif_frames": len(cache.get("mif_indices", [])),
                "union_frames": len(cache.get("union_indices", [])),
                "raw_score_length": len(cache.get("raw_anomaly_scores", [])),
                "union_score_length": len(cache.get("union_scores", [])),
                "sequence_alignment": cache.get("sequence_alignment"),
                "score_alignment": score_alignment["method"],
                "uniform_stride": args.uniform_stride,
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
        "mif_frame_scores_dir": os.path.normpath(args.mif_frame_scores_dir),
        "union_cache_dir": os.path.normpath(args.union_cache_dir),
        "selection": "saved MIF selected frames union uniform stride frames",
        "uniform_stride": args.uniform_stride,
        "projection": args.projection,
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



