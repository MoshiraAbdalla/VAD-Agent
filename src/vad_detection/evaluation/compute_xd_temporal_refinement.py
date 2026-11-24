import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


ANN_ROOT = "data/annotations/xd_test_800.json"
SCORE_FOLDER = "results/predictions/xd_violence_mif_stride64/"
VISION_FOLDER = "results/cached_features/xd_violence_stride64/"


def base_video_key(video_path: str) -> str:
    name = Path(video_path).name
    low = name.lower()
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        if low.endswith(ext):
            return name[: -len(ext)]
    # Keep dotted names as-is (e.g., "Bad.Boys"), do not collapse to "Bad".
    return name


def gaussian_kernel(size, sigma):
    xs = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2 * sigma * sigma))
    return k / (np.sum(k) + 1e-8)


def gaussian_smooth_1d(data, size=15, sigma=10.0):
    if len(data) == 0:
        return np.asarray(data, dtype=np.float32)
    k = gaussian_kernel(size, sigma)
    values = np.asarray(data, dtype=np.float32)
    full = np.convolve(values, k, mode="full")
    offset = (len(k) - 1) // 2
    return full[offset:offset + len(values)]


def gaussian_position_weight(data, sigma):
    data = np.asarray(data, dtype=np.float32)
    n = int(data.size)
    if n == 0:
        return data
    x = np.arange(n, dtype=np.float32)
    mu = float(n // 2)
    s = float(max(1e-6, sigma))
    k = np.exp(-0.5 * ((x - mu) / s) ** 2).astype(np.float32)
    return (k * data).astype(np.float32)


def softmax(v):
    v = np.asarray(v, dtype=np.float32)
    e = np.exp(v - np.max(v))
    return e / (np.sum(e) + 1e-8)


def build_gt(sample):
    gt = np.zeros((int(sample["length"]),), dtype=np.float32)
    labels = sample.get("temporal_label", [])
    for i in range(0, len(labels), 2):
        if labels[i] == -1:
            continue
        s = int(labels[i])
        e = int(min(labels[i + 1], sample["length"]))
        if e > s:
            gt[s:e] = 1.0
    return gt


def normalize_01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-8)


def parse_segments_from_payload(payload):
    # New format: {"segments":[{"start","end","label",...}], ...}
    if isinstance(payload, dict) and isinstance(payload.get("segments", None), list):
        segs = payload.get("segments", [])
        seg_defs = [(int(s.get("start", 0)), int(s.get("end", 0))) for s in segs]
        seg_scores = [float(int(s.get("label", 0))) for s in segs]  # keep hard labels
        seg_anchors = [int(s.get("start", 0)) for s in segs]
        return seg_defs, np.asarray(seg_scores, dtype=np.float32), seg_anchors

    # Legacy score-list format:
    # {"0":{"start":0,"end":150,"score":0}, "16":{...}, ...}
    if isinstance(payload, dict):
        items = []
        for k, v in payload.items():
            if not isinstance(v, dict):
                continue
            try:
                anchor = int(k)
            except Exception:
                continue
            start = int(v.get("start", anchor))
            end = int(v.get("end", anchor))
            score = float(v.get("score", v.get("label", 0.0)))
            items.append((anchor, start, end, score))

        items.sort(key=lambda x: x[0])
        seg_defs = [(it[1], it[2]) for it in items]
        seg_scores = [it[3] for it in items]
        seg_anchors = [it[0] for it in items]
        return seg_defs, np.asarray(seg_scores, dtype=np.float32), seg_anchors

    return [], np.asarray([], dtype=np.float32), []


def infer_sampling_rate(payload, seg_anchors, seg_defs):
    if isinstance(payload, dict):
        stride = int(payload.get("segment_stride", 0) or 0)
        if stride > 0:
            return stride
        seg_len = int(payload.get("segment_len", 0) or 0)
        if seg_len > 0:
            return seg_len

    if len(seg_anchors) > 1:
        diffs = np.diff(np.asarray(seg_anchors, dtype=np.int32))
        med = int(np.median(diffs))
        if med > 0:
            return med

    if len(seg_defs) > 0:
        s0, e0 = seg_defs[0]
        w = int(e0 - s0)
        if w > 0:
            return w

    return 16


def align_features_to_segment_starts(seg_defs, seg_scores, seg_anchors, v_array, payload, length, strict=False):
    """
    Align features to score segments when one side is missing segments.
    Primary target: if score has missing segments but features are full-grid,
    drop the corresponding feature rows by segment start, not by tail truncation.
    """
    n_seg = len(seg_scores)
    n_feat = int(v_array.shape[0])
    if n_seg == n_feat:
        return seg_defs, seg_scores, seg_anchors, v_array, "exact"

    sampling_rate = infer_sampling_rate(payload, seg_anchors, seg_defs)
    expected_starts = list(range(0, int(length), max(1, int(sampling_rate))))
    if len(expected_starts) == 0:
        if strict:
            raise RuntimeError(
                "Strict alignment failed: expected_starts is empty while segment/feature counts mismatch "
                f"(n_seg={n_seg}, n_feat={n_feat}, length={length})."
            )
        n = min(n_seg, n_feat, len(seg_defs))
        return seg_defs[:n], seg_scores[:n], seg_anchors[:n], v_array[:n], "fallback_truncate_empty_expected"

    # Case A: features look full-grid, score has missing starts.
    if n_feat == len(expected_starts) and n_seg <= n_feat:
        by_start = {}
        for i, st in enumerate(expected_starts):
            if st not in by_start:
                by_start[st] = i

        idx = []
        ok = True
        for st in seg_anchors:
            j = by_start.get(int(st), None)
            if j is None:
                ok = False
                break
            idx.append(j)

        if ok and len(idx) == n_seg:
            return seg_defs, seg_scores, seg_anchors, v_array[np.asarray(idx, dtype=np.int64)], "aligned_drop_missing_scores"

    # Case B: scores look full-grid, feature has missing rows.
    if n_seg == len(expected_starts) and n_feat < n_seg:
        # In this scenario, we cannot recover which feature row is missing if
        # feature file has no explicit starts; preserve old behavior but explicit.
        if strict:
            raise RuntimeError(
                "Strict alignment failed: feature rows fewer than full-grid segment count and "
                "cannot infer exact missing feature-row position."
            )
        n = min(n_seg, n_feat, len(seg_defs))
        return seg_defs[:n], seg_scores[:n], seg_anchors[:n], v_array[:n], "fallback_truncate_missing_features_unknown_start"

    if strict:
        raise RuntimeError(
            "Strict alignment failed: score/feature mismatch not resolvable by start-based mapping "
            f"(n_seg={n_seg}, n_feat={n_feat}, expected_starts={len(expected_starts)})."
        )

    n = min(n_seg, n_feat, len(seg_defs))
    return seg_defs[:n], seg_scores[:n], seg_anchors[:n], v_array[:n], "fallback_truncate_generic"


def project_segment_scores_to_frames(score_values, seg_defs, seg_anchors, payload, length, args):
    score_values = np.asarray(score_values, dtype=np.float32)
    length = int(length)
    # Existing plotting/benchmark callers predate the CLI flag.
    gap_policy = getattr(args, "gap_policy", "hold")
    if length <= 0 or len(score_values) != len(seg_defs) or not len(score_values):
        raise ValueError("Projection requires one score per saved segment and a positive video length.")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("Projection received nonfinite scores.")
    if args.projection_mode == "overlap":
        frame_sum = np.zeros((length,), dtype=np.float32)
        frame_cnt = np.zeros((length,), dtype=np.float32)
        for (s, e), sc in zip(seg_defs, score_values):
            s = max(0, min(length, int(s)))
            e = max(s, min(length, int(e)))
            frame_sum[s:e] += float(sc)
            frame_cnt[s:e] += 1.0
        return frame_sum / np.maximum(frame_cnt, 1e-8)

    if int(args.sampling_rate) > 0:
        sampling_rate = int(args.sampling_rate)
    else:
        payload_stride = int(payload.get("segment_stride", 0) or 0) if isinstance(payload, dict) else 0
        payload_seg_len = int(payload.get("segment_len", 0) or 0) if isinstance(payload, dict) else 0
        if payload_stride > 0:
            sampling_rate = payload_stride
        elif payload_seg_len > 0:
            sampling_rate = payload_seg_len
        elif len(seg_anchors) > 1:
            diffs = np.diff(np.asarray(seg_anchors, dtype=np.int32))
            med = int(np.median(diffs))
            sampling_rate = max(1, med)
        else:
            sampling_rate = 16

    # Preserve saved frame starts instead of concatenating rows across gaps.
    starts = [int(start) for start, _ in seg_defs]
    if starts != sorted(set(starts)) or any(start < 0 or start >= length or start % sampling_rate for start in starts):
        raise ValueError("Stride projection requires increasing, unique, in-bounds starts on the stride grid.")
    missing_starts = sorted(set(range(0, length, sampling_rate)) - set(starts))
    if gap_policy not in ("error", "hold", "zero"):
        raise ValueError(f"Unsupported gap policy: {gap_policy}")
    if gap_policy == "error" and missing_starts:
        raise ValueError(f"Missing predictions at starts {missing_starts}; select --gap-policy hold or zero explicitly.")
    pred = np.zeros((length,), dtype=np.float32)
    for i, (start, score) in enumerate(zip(starts, score_values)):
        if gap_policy == "hold":
            end = starts[i + 1] if i + 1 < len(starts) else length
        else:
            end = start + sampling_rate
        pred[start:min(length, end)] = float(score)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-root", default=ANN_ROOT)
    parser.add_argument("--score-folder", default=SCORE_FOLDER)
    parser.add_argument("--vision-folder", default=VISION_FOLDER)
    parser.add_argument(
        "--common-score-folder",
        default="",
        help="If set, evaluate only videos that exist in both --score-folder and this folder.",
    )
    parser.add_argument("--top-ratio", type=float, default=0.15)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--smooth-size", type=int, default=15)
    parser.add_argument("--smooth-sigma", type=float, default=10.0)
    parser.add_argument(
        "--apply-position-weight",
        action="store_true",
        help="Apply frame-level position weighting (Gaussian centered in video).",
    )
    parser.add_argument(
        "--position-weight-sigma",
        type=float,
        default=-1.0,
        help="Sigma for position weighting. If <=0, uses length/2.",
    )
    parser.add_argument(
        "--projection-mode",
        choices=["overlap", "stride"],
        default="overlap",
        help=(
            "overlap: fill by segment [start,end) and average overlaps. "
            "stride: project scores at saved segment starts using the explicit gap policy."
        ),
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=-1,
        help=(
            "Used when --projection-mode stride. "
            "If <1, auto-uses payload['segment_stride'] (fallback payload['segment_len'], then 16)."
        ),
    )
    parser.add_argument(
        "--include-self-sim",
        action="store_true",
        help="Include self similarity in retrieval graph (default excludes self).",
    )
    parser.add_argument(
        "--use-frame-scores",
        action="store_true",
        help="Fuse payload['frame_scores'] with segment-derived frame prediction. Requires --allow-motion-proxy-frame-scores.",
    )
    parser.add_argument(
        "--allow-motion-proxy-frame-scores",
        action="store_true",
        help="Explicitly allow using payload['frame_scores'], which are motion-proxy scores in current MIF runs.",
    )
    parser.add_argument(
        "--frame-score-weight",
        type=float,
        default=0.30,
        help="Fusion weight for frame_scores when --use-frame-scores is enabled.",
    )
    parser.add_argument(
        "--fuse-direct-segment-score",
        action="store_true",
        help="Fuse direct raw segment score projection with temporal refined projection.",
    )
    parser.add_argument(
        "--direct-segment-weight",
        type=float,
        default=0.50,
        help="Weight for direct raw segment projection when --fuse-direct-segment-score is enabled.",
    )
    parser.add_argument(
        "--strict-alignment",
        action="store_true",
        help="Fail fast on any unresolved score/feature mismatch instead of fallback truncation.",
    )
    parser.add_argument(
        "--strict-frame-score-len",
        action="store_true",
        help="When --use-frame-scores, require frame_scores length == video length.",
    )
    parser.add_argument("--only-testing-split", action="store_true")
    parser.add_argument("--testing-videos-root", default="data/XD-Violence-testing/videos/")
    parser.add_argument("--verbose", action="store_true", help="Print per-video processing details.")
    parser.add_argument("--gap-policy", choices=["error", "hold", "zero"], default="error",
                        help="For stride projection: error on absent windows, hold the previous score, or leave missing bins at zero. Initial score is zero.")
    args = parser.parse_args()
    if args.projection_mode == "overlap":
        args.projection_mode = "overlap"
    elif args.projection_mode == "stride":
        args.projection_mode = "stride"
    if args.use_frame_scores and not args.allow_motion_proxy_frame_scores:
        raise RuntimeError(
            "--use-frame-scores is disabled for these experiments because payload['frame_scores'] "
            "are motion-proxy values. Pass --allow-motion-proxy-frame-scores only for a deliberate proxy ablation."
        )

    score_dir = Path(args.score_folder)
    vision_dir = Path(args.vision_folder)
    common_dir = Path(args.common_score_folder) if args.common_score_folder else None

    allowed_keys = None
    if common_dir is not None:
        keys_a = {p.stem for p in score_dir.glob("*.json")}
        keys_b = {p.stem for p in common_dir.glob("*.json")}
        allowed_keys = keys_a & keys_b

    # Accept UTF-8 files with or without BOM.
    with Path(args.ann_root).open(encoding="utf-8-sig") as f:
        annotation = json.load(f)

    seen_keys = {}
    duplicate_keys = {}
    for idx, sample in enumerate(annotation):
        key = base_video_key(sample["video"])
        if key in seen_keys:
            duplicate_keys.setdefault(key, [seen_keys[key]]).append(idx)
        else:
            seen_keys[key] = idx
    if duplicate_keys:
        preview = []
        for key, indices in list(duplicate_keys.items())[:10]:
            labels = [annotation[i].get("label") for i in indices]
            preview.append(f"{key} indices={indices} labels={labels}")
        raise RuntimeError(
            "Duplicate annotation video keys detected after extension stripping. "
            "This would reuse the same prediction file with multiple GT rows, so AUC/AP would be invalid. "
            "Use a deduplicated annotation file such as data/xd_test_800.json. "
            f"duplicates={len(duplicate_keys)} preview={preview}"
        )

    all_pred = []
    all_gt = []
    missing = 0
    used = 0
    fused_count = 0
    aligned_count = 0
    fallback_truncate_count = 0

    for sample in annotation:
        key = base_video_key(sample["video"])
        if args.verbose:
            print(f"Processing video: {sample['video']} (key={key})")
        if args.only_testing_split:
            video_path = Path(args.testing_videos_root) / f"{key}.mp4"
            if not os.path.exists(video_path):
                continue

        if allowed_keys is not None and key not in allowed_keys:
            continue

        score_path = score_dir / f"{key}.json"
        feat_path = vision_dir / f"{key}.npy"
        if (not score_path.exists()) or (not feat_path.exists()):
            missing += 1
            continue

        with score_path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)

        seg_defs, seg_scores, seg_anchors = parse_segments_from_payload(payload)
        if len(seg_defs) == 0:
            missing += 1
            continue

        v_array = np.load(feat_path).astype(np.float32)
        seg_defs, seg_scores, seg_anchors, v_array, align_mode = align_features_to_segment_starts(
            seg_defs=seg_defs,
            seg_scores=seg_scores,
            seg_anchors=seg_anchors,
            v_array=v_array,
            payload=payload,
            length=int(sample["length"]),
            strict=bool(args.strict_alignment),
        )

        n = min(len(seg_scores), v_array.shape[0], len(seg_defs))
        if n == 0:
            missing += 1
            continue

        if align_mode == "aligned_drop_missing_scores":
            aligned_count += 1
        elif align_mode != "exact":
            fallback_truncate_count += 1

        norms = np.linalg.norm(v_array, axis=1, keepdims=True) + 1e-8
        v_norm = v_array / norms
        sim = np.dot(v_norm, v_norm.T)
        if not args.include_self_sim:
            np.fill_diagonal(sim, -1.0)

        if n > 1:
            top_n = max(1, int(float(args.top_ratio) * n))
            top_n = min(top_n, n - 1 if not args.include_self_sim else n)
        else:
            top_n = 1

        v_indices = np.argsort(sim, axis=1)[:, -top_n:]
        v_top_values = np.take_along_axis(sim, v_indices, axis=1)

        refined_seg = []
        for i in range(n):
            neigh = seg_scores[v_indices[i]]
            w = softmax(v_top_values[i] * float(args.temp))
            refined_seg.append(float(np.dot(w, neigh)))

        refined_seg = gaussian_smooth_1d(
            np.asarray(refined_seg, dtype=np.float32),
            size=int(args.smooth_size),
            sigma=float(args.smooth_sigma),
        )

        length = int(sample["length"])
        pred = project_segment_scores_to_frames(refined_seg, seg_defs, seg_anchors, payload, length, args)

        if args.fuse_direct_segment_score:
            direct_pred = project_segment_scores_to_frames(seg_scores[:n], seg_defs[:n], seg_anchors[:n], payload, length, args)
            w_direct = float(max(0.0, min(1.0, args.direct_segment_weight)))
            pred = w_direct * direct_pred + (1.0 - w_direct) * pred

        # Optional fusion with motion proxy frame scores
        if args.use_frame_scores:
            proxy = np.asarray(payload.get("frame_scores", []), dtype=np.float32)
            if args.strict_frame_score_len and proxy.size != length:
                raise RuntimeError(
                    f"Strict frame_scores length check failed for {key}: "
                    f"got={proxy.size}, expected={length}"
                )
            if proxy.size == length:
                w_proxy = float(max(0.0, min(1.0, args.frame_score_weight)))
                proxy = normalize_01(proxy)
                pred = (1.0 - w_proxy) * pred + w_proxy * proxy
                fused_count += 1

        if args.apply_position_weight:
            sigma_pw = float(args.position_weight_sigma)
            if sigma_pw <= 0:
                sigma_pw = float(max(1.0, length / 2.0))
            pred = gaussian_position_weight(pred, sigma=sigma_pw)
        # print("pred",pred[0:200])
        gt = build_gt(sample)
        all_pred.extend(pred.tolist())

        all_gt.extend(gt.tolist())
        used += 1

    if len(all_gt) == 0 or len(all_pred) == 0:
        raise RuntimeError("No valid predictions available for AUC.")

    fpr, tpr, _ = roc_curve(all_gt, all_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_gt, all_pred)
    pr_auc = auc(recall, precision)
    print(f"AUC={roc_auc:.9f}")
    print(f"AP={pr_auc:.9f}")
    print(f"videos_used={used}")
    if allowed_keys is not None:
        print(f"common_subset_videos={len(allowed_keys)}")
    print(f"missing_videos={missing}")
    print(
        f"params: top_ratio={args.top_ratio} temp={args.temp} "
        f"smooth_size={args.smooth_size} smooth_sigma={args.smooth_sigma} "
        f"apply_position_weight={args.apply_position_weight} "
        f"position_weight_sigma={args.position_weight_sigma} "
        f"projection_mode={args.projection_mode} sampling_rate={args.sampling_rate} gap_policy={args.gap_policy} "
        f"fuse_direct_segment_score={args.fuse_direct_segment_score} "
        f"direct_segment_weight={args.direct_segment_weight} "
        f"exclude_self={not args.include_self_sim} "
        f"use_frame_scores={args.use_frame_scores} frame_score_weight={args.frame_score_weight} "
        f"only_testing_split={args.only_testing_split} "
        f"strict_alignment={args.strict_alignment} strict_frame_score_len={args.strict_frame_score_len}"
    )
    if args.use_frame_scores:
        print(f"videos_fused_with_frame_scores={fused_count}")
    print(f"alignment_by_start_videos={aligned_count}")
    print(f"fallback_truncate_videos={fallback_truncate_count}")


if __name__ == "__main__":
    main()







