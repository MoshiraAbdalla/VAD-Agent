import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def load_eval_module(dataset):
    if dataset == "ucf":
        import compute_ucf_temporal_refinement as eval_mod

        return eval_mod
    if dataset == "xd":
        import compute_xd_temporal_refinement as eval_mod

        return eval_mod
    raise ValueError(f"Unsupported dataset: {dataset}")


def project_segment_scores(seg_defs, seg_scores, seg_anchors, payload, length, projection_mode, sampling_rate, gap_policy="error"):
    length = int(length)
    seg_scores = np.asarray(seg_scores, dtype=np.float32)
    if length <= 0 or len(seg_scores) != len(seg_defs) or not len(seg_scores):
        raise ValueError("Projection requires one score per saved segment and a positive video length.")
    if not np.all(np.isfinite(seg_scores)):
        raise ValueError("Projection received nonfinite scores.")

    if projection_mode == "overlap":
        frame_sum = np.zeros((length,), dtype=np.float32)
        frame_cnt = np.zeros((length,), dtype=np.float32)
        for (s, e), sc in zip(seg_defs, seg_scores):
            s = max(0, min(length, int(s)))
            e = max(s, min(length, int(e)))
            frame_sum[s:e] += float(sc)
            frame_cnt[s:e] += 1.0
        return frame_sum / np.maximum(frame_cnt, 1e-8)

    if sampling_rate > 0:
        stride = int(sampling_rate)
    else:
        payload_stride = int(payload.get("segment_stride", 0) or 0) if isinstance(payload, dict) else 0
        payload_seg_len = int(payload.get("segment_len", 0) or 0) if isinstance(payload, dict) else 0
        if payload_stride > 0:
            stride = payload_stride
        elif payload_seg_len > 0:
            stride = payload_seg_len
        elif len(seg_anchors) > 1:
            stride = max(1, int(np.median(np.diff(np.asarray(seg_anchors, dtype=np.int32)))))
        else:
            stride = 16

    # Preserve saved frame starts instead of concatenating rows across gaps.
    starts = [int(start) for start, _ in seg_defs]
    if starts != sorted(set(starts)) or any(start < 0 or start >= length or start % stride for start in starts):
        raise ValueError("Stride projection requires increasing, unique, in-bounds starts on the stride grid.")
    missing_starts = sorted(set(range(0, length, stride)) - set(starts))
    if gap_policy not in ("error", "hold", "zero"):
        raise ValueError(f"Unsupported gap policy: {gap_policy}")
    if gap_policy == "error" and missing_starts:
        raise ValueError(f"Missing predictions at starts {missing_starts}; select --gap-policy hold or zero explicitly.")
    pred = np.zeros((length,), dtype=np.float32)
    for i, (start, score) in enumerate(zip(starts, seg_scores)):
        if gap_policy == "hold":
            end = starts[i + 1] if i + 1 < len(starts) else length
        else:
            end = start + stride
        pred[start:min(length, end)] = float(score)
    return pred


def sample_key(eval_mod, dataset, video):
    if dataset == "xd":
        return eval_mod.base_video_key(video)
    return eval_mod.key_from_video_path(video)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw segment-level model scores projected to frames. "
            "This skips feature-neighbor temporal refinement and Gaussian segment smoothing."
        )
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--ann-root", required=True)
    parser.add_argument("--score-folder", required=True)
    parser.add_argument(
        "--vision-folder",
        default="",
        help="Optional. If set, use only videos with matching feature files and apply the same score/feature alignment as the refined evaluator.",
    )
    parser.add_argument(
        "--common-score-folder",
        default="",
        help="If set, evaluate only videos that exist in both --score-folder and this folder.",
    )
    parser.add_argument("--projection-mode", choices=["overlap", "stride"], default="overlap")
    parser.add_argument("--sampling-rate", type=int, default=-1)
    parser.add_argument(
        "--apply-position-weight",
        action="store_true",
        help="Optional final frame-level position weighting. Do not use for strict binary 0/1 direct scores.",
    )
    parser.add_argument("--position-weight-sigma", type=float, default=-1.0)
    parser.add_argument("--strict-alignment", action="store_true")
    parser.add_argument("--gap-policy", choices=["error", "hold", "zero"], default="error",
                        help="For stride projection: error on absent windows, hold the previous score, or leave missing bins at zero. Initial score is zero.")
    args = parser.parse_args()

    eval_mod = load_eval_module(args.dataset)
    score_dir = Path(args.score_folder)
    vision_dir = Path(args.vision_folder) if args.vision_folder else None
    common_dir = Path(args.common_score_folder) if args.common_score_folder else None

    allowed_keys = None
    if common_dir is not None:
        allowed_keys = {p.stem for p in score_dir.glob("*.json")} & {p.stem for p in common_dir.glob("*.json")}

    with Path(args.ann_root).open(encoding="utf-8-sig") as f:
        annotation = json.load(f)

    all_pred = []
    all_gt = []
    used = 0
    missing = 0
    aligned_count = 0
    fallback_truncate_count = 0

    for sample in annotation:
        key = sample_key(eval_mod, args.dataset, sample["video"])
        if allowed_keys is not None and key not in allowed_keys:
            continue

        score_path = score_dir / f"{key}.json"
        if not score_path.exists():
            missing += 1
            continue

        feat_path = vision_dir / f"{key}.npy" if vision_dir is not None else None
        if feat_path is not None and not feat_path.exists():
            missing += 1
            continue

        with score_path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)

        seg_defs, seg_scores, seg_anchors = eval_mod.parse_segments_from_payload(payload)
        if len(seg_defs) == 0:
            missing += 1
            continue

        if feat_path is not None:
            v_array = np.load(feat_path).astype(np.float32)
            seg_defs, seg_scores, seg_anchors, _, align_mode = eval_mod.align_features_to_segment_starts(
                seg_defs=seg_defs,
                seg_scores=seg_scores,
                seg_anchors=seg_anchors,
                v_array=v_array,
                payload=payload,
                length=int(sample["length"]),
                strict=bool(args.strict_alignment),
            )
            if align_mode == "aligned_drop_missing_scores":
                aligned_count += 1
            elif align_mode != "exact":
                fallback_truncate_count += 1

        pred = project_segment_scores(
            seg_defs=seg_defs,
            seg_scores=seg_scores,
            seg_anchors=seg_anchors,
            payload=payload,
            length=int(sample["length"]),
            projection_mode=args.projection_mode,
            sampling_rate=int(args.sampling_rate),
            gap_policy=args.gap_policy,
        )

        if args.apply_position_weight:
            sigma_pw = float(args.position_weight_sigma)
            if sigma_pw <= 0:
                sigma_pw = float(max(1.0, int(sample["length"]) / 2.0))
            pred = eval_mod.gaussian_position_weight(pred, sigma=sigma_pw)

        gt = eval_mod.build_gt(sample)
        all_pred.extend(pred.tolist())
        all_gt.extend(gt.tolist())
        used += 1

    if len(all_gt) == 0 or len(all_pred) == 0:
        raise RuntimeError("No valid predictions available for direct segment metric.")

    fpr, tpr, _ = roc_curve(all_gt, all_pred)
    roc_auc = auc(fpr, tpr)
    print(f"AUC={roc_auc:.9f}")
    if args.dataset == "xd":
        precision, recall, _ = precision_recall_curve(all_gt, all_pred)
        pr_auc = auc(recall, precision)
        print(f"AP={pr_auc:.9f}")
    print(f"videos_used={used}")
    if allowed_keys is not None:
        print(f"common_subset_videos={len(allowed_keys)}")
    print(f"missing_videos={missing}")
    print(
        f"params: dataset={args.dataset} projection_mode={args.projection_mode} "
        f"sampling_rate={args.sampling_rate} gap_policy={args.gap_policy} apply_position_weight={args.apply_position_weight} "
        f"position_weight_sigma={args.position_weight_sigma} strict_alignment={args.strict_alignment}"
    )
    if vision_dir is not None:
        print(
            f"alignment_summary: aligned_drop_missing_scores={aligned_count} "
            f"fallback_truncate={fallback_truncate_count}"
        )


if __name__ == "__main__":
    main()


