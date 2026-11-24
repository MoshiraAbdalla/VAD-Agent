#!/usr/bin/env python3
"""Compute HIVAU AUC/AP using MIF over Holmes/InternVL CLS features.

This is the feature-aligned MIF variant:

  1. Split video into 160-frame chunks.
  2. Encode every frame in the chunk with HolmesVAU `model.vision_model`.
  3. Extract the frame-level CLS feature: `last_hidden_state[:, 0, :]`.
  4. Run MIF local-minimum selection on those Holmes CLS features.
  5. Feed the selected Holmes CLS sequence directly into the ATS anomaly scorer.
  6. Project selected scores back to per-frame scores and compute direct AUC/AP.

It does not use the VAD-Agent GIT selector or saved VAD-Agent MIF indices.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.nn.functional import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from holmesvau.holmesvau_utils import get_pixel_values, load_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class _ArrayFrame:
    def __init__(self, array):
        self._array = array

    def asnumpy(self):
        return self._array


def _frame_sort_key(path):
    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    return stem


class FrameFolderReader:
    """Minimal Decord-like reader for extracted frame folders."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.frames = sorted(
            [p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
            key=_frame_sort_key,
        )
        if not self.frames:
            raise ValueError(f"No image frames found in {self.folder}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        image = Image.open(self.frames[int(index)]).convert("RGB")
        return _ArrayFrame(np.asarray(image))


def open_video_or_frame_folder(path):
    path_obj = Path(path)
    if path_obj.is_dir():
        return FrameFolderReader(path_obj)
    return VideoReader(str(path_obj), ctx=cpu(0), num_threads=1)


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path, data):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def video_key_from_prediction(row):
    return Path(row.get("video_id") or row.get("video_name") or row.get("video_path") or "").stem


def build_gt_copy(mdf_gt_scores):
    copied = {
        "name": mdf_gt_scores.get("name"),
        "description": "Copied GT arrays from the released HIVAU overlap-150 evaluation artifact.",
        "parts": {},
    }
    for part_name, part in mdf_gt_scores.get("parts", {}).items():
        videos = []
        for video in part.get("videos", []):
            videos.append(
                {
                    "dataset": video.get("dataset") or part_name,
                    "part": part_name,
                    "video": video["video"],
                    "length": int(video.get("length", len(video.get("gt", [])))),
                    "temporal_label": video.get("temporal_label"),
                    "gt": video.get("gt", []),
                }
            )
        copied["parts"][part_name] = {"dataset": part.get("dataset") or part_name, "videos": videos}
    return copied


def flatten_gt_videos(gt_copy):
    flat = []
    for part_name, part in gt_copy.get("parts", {}).items():
        for video in part.get("videos", []):
            item = dict(video)
            item["part"] = part_name
            flat.append(item)
    return flat


def metric_pair(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.size == 0 or y_score.size == 0 or len(np.unique(y_true)) < 2:
        return {"AUC": None, "AP": None, "reason": "labels contain only one class"}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "AUC": float(roc_auc_score(y_true, y_score)),
            "AP": float(average_precision_score(y_true, y_score)),
            "metric_backend": "sklearn",
        }
    except Exception:
        return {
            "AUC": fallback_roc_auc(y_true, y_score),
            "AP": fallback_average_precision(y_true, y_score),
            "metric_backend": "fallback_rank",
        }


def fallback_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    return float((original_ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fallback_average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    desc = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[desc]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / np.arange(1, len(sorted_true) + 1, dtype=np.float64)
    n_pos = int((sorted_true == 1).sum())
    return float(precision[sorted_true == 1].sum() / max(1, n_pos))


@torch.inference_mode()
def encode_holmes_cls(vr, frame_indices, model, batch_size):
    cls_chunks = []
    for start in range(0, len(frame_indices), batch_size):
        batch_indices = [int(i) for i in frame_indices[start : start + batch_size]]
        pixel_values, _ = get_pixel_values(vr, batch_indices)
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
        vit_embeds = model.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True).last_hidden_state
        cls_chunks.append(vit_embeds[:, 0, :].detach().float().cpu())
        del pixel_values, vit_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(cls_chunks, dim=0) if cls_chunks else torch.empty((0, 1024), dtype=torch.float32)


def select_mif_from_cls_features(features, k, w, alpha=2.5):
    if features.numel() == 0:
        return []
    n = int(features.shape[0])
    k = max(1, min(int(k), n))
    if n <= k:
        return list(range(n))
    if w == -1:
        w = max(1, int(n // (alpha * k)))
    w = max(1, int(w))

    feats = normalize(features.float(), dim=1)
    sims = feats @ feats.transpose(0, 1)

    lcl_avg = torch.full((n,), float("inf"), dtype=torch.float32)
    valid_start, valid_end = (w, n - w) if n > 2 * w else (0, n)
    for i in range(valid_start, valid_end):
        left = max(0, i - w)
        right = min(n, i + w + 1)
        denom = max(1, right - left - 1)
        lcl_avg[i] = (sims[i, left:right].sum().float() - 1.0) / float(denom)

    selected = []
    blocked = torch.zeros((n,), dtype=torch.bool)
    finite = torch.isfinite(lcl_avg)
    while len(selected) < k:
        candidates = finite & (~blocked)
        if not torch.any(candidates):
            candidates = ~blocked
        if not torch.any(candidates):
            break
        masked = torch.where(candidates, lcl_avg, torch.full_like(lcl_avg, float("inf")))
        idx = int(torch.argmin(masked).item())
        if not np.isfinite(float(masked[idx])):
            remaining = torch.where(~blocked)[0]
            if remaining.numel() == 0:
                break
            idx = int(remaining[0].item())
        selected.append(idx)
        left = max(0, idx - w)
        right = min(n, idx + w + 1)
        blocked[left:right] = True

    if len(selected) < k:
        for idx in range(n):
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= k:
                    break

    return sorted(set(int(i) for i in selected[:k]))


@torch.inference_mode()
def select_and_score_video(video_path, model, sampler, segment_len, segment_stride, dense_equivalent_stride, mif_w, batch_size):
    vr = open_video_or_frame_folder(video_path)
    n_frames = len(vr)
    segments = []
    selected_indices = []
    selected_feature_chunks = []

    for start in range(0, n_frames, segment_stride):
        end = min(start + segment_len, n_frames)
        if end <= start:
            continue
        frame_ids = list(range(start, end))
        k = max(1, int(math.ceil(len(frame_ids) / float(dense_equivalent_stride))))
        k = min(k, len(frame_ids))
        cls_features = encode_holmes_cls(vr, frame_ids, model, batch_size)
        local_indices = select_mif_from_cls_features(cls_features, k=k, w=mif_w)
        global_indices = [int(frame_ids[i]) for i in local_indices]
        segments.append(
            {
                "start": int(start),
                "end": int(end),
                "k": int(k),
                "mif_indices": global_indices,
            }
        )
        selected_indices.extend(global_indices)
        if local_indices:
            selected_feature_chunks.append(cls_features[local_indices])
        del cls_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if selected_feature_chunks:
        selected_features = torch.cat(selected_feature_chunks, dim=0)
        vid_feats = selected_features.unsqueeze(0).to(sampler.device)
        selected_scores = sampler.anomaly_scorer(vid_feats)["anomaly_scores"][0].detach().cpu().numpy().astype(np.float32)
    else:
        selected_scores = np.asarray([], dtype=np.float32)

    return n_frames, segments, selected_indices, selected_scores


def project_segments(length, segments, score_by_frame, mode):
    frame_sum = np.zeros((length,), dtype=np.float32)
    frame_count = np.zeros((length,), dtype=np.float32)
    for segment in segments:
        start = max(0, int(segment["start"]))
        end = min(length, int(segment["end"]))
        selected = [idx for idx in segment.get("mif_indices", []) if start <= idx < end and idx in score_by_frame]
        if end <= start or not selected:
            continue
        selected = sorted(selected)
        selected_scores = np.asarray([score_by_frame[idx] for idx in selected], dtype=np.float32)
        if mode == "segment_mean":
            projected = np.full((end - start,), float(selected_scores.mean()), dtype=np.float32)
        else:
            frames = np.arange(start, end, dtype=np.int32)
            selected_arr = np.asarray(selected, dtype=np.int32)
            if mode == "linear":
                projected = np.interp(frames, selected_arr, selected_scores, left=float(selected_scores[0]), right=float(selected_scores[-1]))
                projected = projected.astype(np.float32)
            elif mode == "nearest":
                pos = np.searchsorted(selected_arr, frames)
                left = np.clip(pos - 1, 0, len(selected_arr) - 1)
                right = np.clip(pos, 0, len(selected_arr) - 1)
                choose_right = np.abs(selected_arr[right] - frames) < np.abs(frames - selected_arr[left])
                nearest = np.where(choose_right, right, left)
                projected = selected_scores[nearest].astype(np.float32)
            elif mode.startswith("expmax"):
                tau = float(mode.replace("expmax", ""))
                distances = np.abs(frames.astype(np.float32)[:, None] - selected_arr.astype(np.float32)[None, :])
                projected = np.max(selected_scores[None, :] * np.exp(-distances / tau), axis=1).astype(np.float32)
            else:
                raise ValueError(f"Unknown projection mode: {mode}")
        frame_sum[start:end] += projected
        frame_count[start:end] += 1.0
    frame_scores = np.zeros((length,), dtype=np.float32)
    valid = frame_count > 0
    frame_scores[valid] = frame_sum[valid] / np.maximum(frame_count[valid], 1e-8)
    return frame_scores, int(valid.sum())


def build_projection_ranges(length, segments, score_by_frame, mode):
    projection_ranges = []
    for segment in segments:
        start = max(0, int(segment["start"]))
        end = min(length, int(segment["end"]))
        selected = sorted(idx for idx in segment.get("mif_indices", []) if start <= idx < end and idx in score_by_frame)
        ranges = []
        if selected and mode == "nearest":
            for i, source_frame in enumerate(selected):
                frame_start = start if i == 0 else (selected[i - 1] + source_frame) // 2 + 1
                frame_end = end if i == len(selected) - 1 else (source_frame + selected[i + 1]) // 2 + 1
                ranges.append(
                    {
                        "frame_start": int(frame_start),
                        "frame_end_exclusive": int(frame_end),
                        "source_frame": int(source_frame),
                        "score": float(score_by_frame[source_frame]),
                    }
                )
        projection_ranges.append({"start": int(start), "end": int(end), "projection": mode, "ranges": ranges})
    return projection_ranges


def main():
    parser = argparse.ArgumentParser(description="Holmes CLS MIF-160 frame selection + Holmes ATS direct HIVAU AUC/AP.")
    parser.add_argument("--ckpt", default="HolmesVAU-2B")
    parser.add_argument("--sampler_path", default=os.path.join("holmesvau", "ATS", "anomaly_scorer.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions", default="evaluation/hivau_video_predictions.json")
    parser.add_argument(
        "--mdf_gt_scores",
        default="data/annotations/hivau150_overlap150_gt_scores_auc.json",
    )
    parser.add_argument("--gt_copy_out", default="agentic_inference/evaluation/outputs/hivau150_overlap150_gt_copy.json")
    parser.add_argument("--out", default="agentic_inference/evaluation/outputs/hivau_video_holmescls_mif160_ats_direct_auc.json")
    parser.add_argument("--frame_scores_dir", default="agentic_inference/evaluation/outputs/holmescls_mif160_ats_frame_scores")
    parser.add_argument("--segment_len", type=int, default=160)
    parser.add_argument("--segment_stride", type=int, default=160)
    parser.add_argument("--dense_equivalent_stride", type=int, default=16)
    parser.add_argument("--mif_w", type=int, default=8)
    parser.add_argument("--projection", choices=["nearest", "linear", "segment_mean", "expmax64", "expmax128"], default="nearest")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mdf = load_json(args.mdf_gt_scores)
    gt_copy = build_gt_copy(mdf)
    save_json(args.gt_copy_out, gt_copy)
    gt_videos = flatten_gt_videos(gt_copy)
    if args.max_videos and args.max_videos > 0:
        gt_videos = gt_videos[: args.max_videos]

    predictions = load_json(args.predictions)
    pred_by_key = {video_key_from_prediction(row): row for row in predictions}

    print("Loading HolmesVAU + ATS...", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer, generation_config, sampler = load_model(args.ckpt, args.sampler_path, device)
    sampler.anomaly_scorer.eval()

    frame_scores_dir = Path(args.frame_scores_dir)
    frame_scores_dir.mkdir(parents=True, exist_ok=True)

    all_gt = []
    all_scores = []
    per_video = []
    missing = []

    for idx, gt_video in enumerate(gt_videos, start=1):
        key = gt_video["video"]
        row = pred_by_key.get(key)
        if row is None:
            missing.append({"video": key, "reason": "prediction not found"})
            continue
        video_path = row.get("video_path")
        if not video_path or not os.path.exists(video_path):
            missing.append({"video": key, "reason": f"video path not found: {video_path}"})
            continue

        cache_path = frame_scores_dir / f"{gt_video['part']}_{key}.json"
        if args.resume and cache_path.exists():
            cache = load_json(cache_path)
            frame_scores = np.asarray(cache["frame_scores"], dtype=np.float32)
            segments = cache.get("segments", [])
            selected_indices = (cache.get("selected_mif_indices") or cache.get("selected_maf_indices") or [])
            covered_frames = int(cache.get("covered_frames", int(np.count_nonzero(frame_scores))))
        else:
            print(f"[{idx}/{len(gt_videos)}] {gt_video['part']}/{key}: Holmes-CLS MIF-160 selection", flush=True)
            length, segments, selected_indices, selected_scores = select_and_score_video(
                video_path=video_path,
                model=model,
                sampler=sampler,
                segment_len=args.segment_len,
                segment_stride=args.segment_stride,
                dense_equivalent_stride=args.dense_equivalent_stride,
                mif_w=args.mif_w,
                batch_size=args.batch_size,
            )
            print(f"[{idx}/{len(gt_videos)}] {gt_video['part']}/{key}: selected_mif_frames={len(selected_indices)}", flush=True)
            score_by_frame = {int(frame_idx): float(score) for frame_idx, score in zip(selected_indices, selected_scores)}
            frame_scores, covered_frames = project_segments(length, segments, score_by_frame, args.projection)
            cache = {
                "dataset": gt_video["dataset"],
                "part": gt_video["part"],
                "video": key,
                "video_path": video_path,
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

        gt = np.asarray(gt_video["gt"], dtype=np.uint8)
        n = min(len(gt), len(frame_scores))
        gt_eval = gt[:n]
        score_eval = frame_scores[:n]
        metrics = metric_pair(gt_eval, score_eval)
        per_video.append(
            {
                "dataset": gt_video["dataset"],
                "part": gt_video["part"],
                "video": key,
                "length": int(n),
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
    parts = {}
    for part_name in sorted({item["part"] for item in per_video}):
        indices = [i for i, item in enumerate(per_video) if item["part"] == part_name]
        part_gt = np.concatenate([all_gt[i] for i in indices])
        part_score = np.concatenate([all_scores[i] for i in indices])
        parts[part_name] = {
            "videos": len(indices),
            "frames": int(part_gt.size),
            "positive_frames": int(part_gt.sum()),
            **metric_pair(part_gt, part_score),
        }

    valid_auc = [item["AUC"] for item in per_video if item.get("AUC") is not None]
    valid_ap = [item["AP"] for item in per_video if item.get("AP") is not None]
    output = {
        "mdf_gt_scores": os.path.normpath(args.mdf_gt_scores),
        "gt_copy_out": os.path.normpath(args.gt_copy_out),
        "predictions": os.path.normpath(args.predictions),
        "frame_scores_dir": os.path.normpath(args.frame_scores_dir),
        "feature_space": "HolmesVAU model.vision_model CLS token",
        "segment_len": args.segment_len,
        "segment_stride": args.segment_stride,
        "dense_equivalent_stride": args.dense_equivalent_stride,
        "mif_w": args.mif_w,
        "projection": args.projection,
        "num_videos_evaluated": len(per_video),
        "num_videos_missing": len(missing),
        "total_frames": int(all_gt_arr.size),
        "positive_frames": int(all_gt_arr.sum()),
        "overall": metric_pair(all_gt_arr, all_score_arr),
        "parts": parts,
        "macro_per_video": {
            "AUC": None if not valid_auc else float(np.mean(valid_auc)),
            "AP": None if not valid_ap else float(np.mean(valid_ap)),
            "num_videos": len(valid_auc),
        },
        "per_video": per_video,
        "missing": missing,
    }
    save_json(args.out, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




