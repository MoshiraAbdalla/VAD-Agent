import argparse
import json
from pathlib import Path

import numpy as np


def strip_video_extension(video_name):
    name = str(video_name).replace("\\", "/").split("/")[-1]
    lower = name.lower()
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name


def load_video_list(path):
    names = set()
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            names.add(strip_video_extension(name))
    return names


def build_gt(sample):
    length = int(sample["length"])
    gt = np.zeros((length,), dtype=np.float32)
    labels = sample.get("temporal_label", [])
    for i in range(0, len(labels), 2):
        if labels[i] == -1:
            continue
        start = max(0, min(length, int(labels[i])))
        end = max(start, min(length, int(labels[i + 1])))
        if end > start:
            gt[start:end] = 1.0
    return gt


def gaussian_kernel(size, sigma):
    xs = np.linspace(-(size // 2), size // 2, int(size), dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
    return kernel / (np.sum(kernel) + 1e-8)


def gaussian_smooth_1d(data, size=15, sigma=10.0):
    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        return data
    kernel = gaussian_kernel(size, sigma)
    full = np.convolve(data, kernel, mode="full")
    offset = (len(kernel) - 1) // 2
    return full[offset:offset + len(data)].astype(np.float32)


def gaussian_position_weight(data, sigma):
    data = np.asarray(data, dtype=np.float32)
    n = int(data.size)
    if n == 0:
        return data
    x = np.arange(n, dtype=np.float32)
    mu = float(n // 2)
    sigma = float(max(1e-6, sigma))
    weights = np.exp(-0.5 * ((x - mu) / sigma) ** 2).astype(np.float32)
    return (weights * data).astype(np.float32)


def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    exp = np.exp(values - np.max(values))
    return exp / (np.sum(exp) + 1e-8)


def parse_segments_from_payload(payload):
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        segments = payload.get("segments") or []
        seg_defs = [(int(s.get("start", 0)), int(s.get("end", 0))) for s in segments]
        seg_scores = [float(int(s.get("label", 0))) for s in segments]
        seg_anchors = [int(s.get("start", 0)) for s in segments]
        return seg_defs, np.asarray(seg_scores, dtype=np.float32), seg_anchors

    if isinstance(payload, dict):
        items = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            try:
                anchor = int(key)
            except Exception:
                continue
            start = int(value.get("start", anchor))
            end = int(value.get("end", anchor))
            score = float(value.get("score", value.get("label", 0.0)))
            items.append((anchor, start, end, score))
        items.sort(key=lambda item: item[0])
        return (
            [(item[1], item[2]) for item in items],
            np.asarray([item[3] for item in items], dtype=np.float32),
            [item[0] for item in items],
        )

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
        median = int(np.median(diffs))
        if median > 0:
            return median
    if seg_defs:
        width = int(seg_defs[0][1] - seg_defs[0][0])
        if width > 0:
            return width
    return 16


def align_features_to_segment_starts(seg_defs, seg_scores, seg_anchors, features, payload, length, strict=False):
    n_seg = len(seg_scores)
    n_feat = int(features.shape[0])
    if n_seg == n_feat:
        return seg_defs, seg_scores, seg_anchors, features, "exact"

    sampling_rate = infer_sampling_rate(payload, seg_anchors, seg_defs)
    expected_starts = list(range(0, int(length), max(1, int(sampling_rate))))
    if n_feat == len(expected_starts) and n_seg <= n_feat:
        by_start = {start: idx for idx, start in enumerate(expected_starts)}
        indices = []
        ok = True
        for start in seg_anchors:
            idx = by_start.get(int(start))
            if idx is None:
                ok = False
                break
            indices.append(idx)
        if ok and len(indices) == n_seg:
            return seg_defs, seg_scores, seg_anchors, features[np.asarray(indices, dtype=np.int64)], "aligned_drop_missing_scores"

    if strict:
        raise RuntimeError(f"Unresolved feature/score mismatch: n_seg={n_seg}, n_feat={n_feat}, length={length}")

    n = min(n_seg, n_feat, len(seg_defs))
    return seg_defs[:n], seg_scores[:n], seg_anchors[:n], features[:n], "fallback_truncate"


def refine_segments(seg_scores, features, top_ratio, temp, include_self_sim):
    n = min(len(seg_scores), int(features.shape[0]))
    if n == 0:
        return np.asarray([], dtype=np.float32)
    seg_scores = np.asarray(seg_scores[:n], dtype=np.float32)
    features = np.asarray(features[:n], dtype=np.float32)
    norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norm
    sim = np.dot(features, features.T)
    if not include_self_sim:
        np.fill_diagonal(sim, -1.0)
    if n > 1:
        top_n = max(1, int(float(top_ratio) * n))
        top_n = min(top_n, n if include_self_sim else n - 1)
    else:
        top_n = 1
    indices = np.argsort(sim, axis=1)[:, -top_n:]
    values = np.take_along_axis(sim, indices, axis=1)
    refined = []
    for idx in range(n):
        weights = softmax(values[idx] * float(temp))
        refined.append(float(np.dot(weights, seg_scores[indices[idx]])))
    return np.asarray(refined, dtype=np.float32)


def project_segments_to_frames(seg_defs, seg_anchors, segment_scores, payload, length, projection_mode, sampling_rate_arg, gap_policy="error"):
    length = int(length)
    segment_scores = np.asarray(segment_scores, dtype=np.float32)
    if length <= 0 or len(segment_scores) != len(seg_defs) or not len(segment_scores):
        raise ValueError("Projection requires one score per saved segment and a positive video length.")
    if not np.all(np.isfinite(segment_scores)):
        raise ValueError("Projection received nonfinite scores.")
    if projection_mode == "overlap":
        frame_sum = np.zeros((length,), dtype=np.float32)
        frame_count = np.zeros((length,), dtype=np.float32)
        for (start, end), score in zip(seg_defs, segment_scores):
            start = max(0, min(length, int(start)))
            end = max(start, min(length, int(end)))
            frame_sum[start:end] += float(score)
            frame_count[start:end] += 1.0
        return frame_sum / np.maximum(frame_count, 1e-8)

    if int(sampling_rate_arg) > 0:
        sampling_rate = int(sampling_rate_arg)
    else:
        sampling_rate = infer_sampling_rate(payload, seg_anchors, seg_defs)

    starts = [int(start) for start, _ in seg_defs]
    if starts != sorted(set(starts)) or any(start < 0 or start >= length or start % sampling_rate for start in starts):
        raise ValueError("Stride projection requires increasing, unique, in-bounds starts on the stride grid.")
    missing_starts = sorted(set(range(0, length, sampling_rate)) - set(starts))
    if gap_policy not in ("error", "hold", "zero"):
        raise ValueError(f"Unsupported gap policy: {gap_policy}")
    if gap_policy == "error" and missing_starts:
        raise ValueError(f"Missing predictions at starts {missing_starts}; select --gap-policy hold or zero explicitly.")
    pred = np.zeros(length, dtype=np.float32)
    for idx, (start, score) in enumerate(zip(starts, segment_scores)):
        if gap_policy == "hold":
            end = starts[idx + 1] if idx + 1 < len(starts) else length
        else:
            end = start + sampling_rate
        pred[start:min(length, end)] = float(score)
    return pred



def binary_auc_rank(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    idx = 0
    while idx < len(sorted_scores):
        end = idx + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[idx]:
            end += 1
        ranks[idx:end] = (idx + 1 + end) / 2.0
        idx = end
    original = np.empty_like(ranks)
    original[order] = ranks
    rank_sum_pos = float(np.sum(original[pos]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def pr_auc_trapezoid(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(n_pos)
    change = np.r_[np.diff(recall) != 0, True]
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(np.r_[1.0, precision[change]], np.r_[0.0, recall[change]]))


def evaluate(args):
    subset = load_video_list(args.video_list)
    score_dir = Path(args.score_folder)
    feature_dir = Path(args.vision_folder)

    with Path(args.ann_root).open("r", encoding="utf-8-sig") as f:
        annotations = json.load(f)

    all_gt = []
    all_pred = []
    used = 0
    missing = []
    aligned = 0
    truncated = 0

    for sample in annotations:
        key = strip_video_extension(sample["video"])
        if key not in subset:
            continue
        score_path = score_dir / f"{key}.json"
        feature_path = feature_dir / f"{key}.npy"
        if not score_path.exists() or not feature_path.exists():
            missing.append(key)
            continue
        with score_path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        seg_defs, seg_scores, seg_anchors = parse_segments_from_payload(payload)
        if not seg_defs:
            missing.append(key)
            continue
        features = np.load(feature_path).astype(np.float32)
        seg_defs, seg_scores, seg_anchors, features, align_mode = align_features_to_segment_starts(
            seg_defs, seg_scores, seg_anchors, features, payload, int(sample["length"]), strict=bool(args.strict_alignment)
        )
        if align_mode == "aligned_drop_missing_scores":
            aligned += 1
        elif align_mode != "exact":
            truncated += 1
        n = min(len(seg_defs), len(seg_scores), len(seg_anchors), int(features.shape[0]))
        if n == 0:
            missing.append(key)
            continue
        seg_defs = seg_defs[:n]
        seg_scores = seg_scores[:n]
        seg_anchors = seg_anchors[:n]
        features = features[:n]
        refined = refine_segments(seg_scores, features, args.top_ratio, args.temp, bool(args.include_self_sim))
        refined = gaussian_smooth_1d(refined, size=args.smooth_size, sigma=args.smooth_sigma)
        pred = project_segments_to_frames(
            seg_defs, seg_anchors, refined, payload, int(sample["length"]), args.projection_mode, args.sampling_rate, args.gap_policy
        )
        if args.fuse_direct_segment_score:
            w_direct = float(args.direct_segment_weight)
            if w_direct < 0.0 or w_direct > 1.0:
                raise ValueError("--direct-segment-weight must be in [0, 1]")
            direct_pred = project_segments_to_frames(
                seg_defs, seg_anchors, seg_scores, payload, int(sample["length"]), args.projection_mode, args.sampling_rate, args.gap_policy
            )
            pred = w_direct * direct_pred + (1.0 - w_direct) * pred
        if args.apply_position_weight:
            sigma = float(args.position_weight_sigma)
            if sigma <= 0:
                sigma = float(max(1.0, int(sample["length"]) / 2.0))
            pred = gaussian_position_weight(pred, sigma=sigma)
        all_pred.extend(pred.tolist())
        all_gt.extend(build_gt(sample).tolist())
        used += 1

    auc = binary_auc_rank(all_gt, all_pred)
    ap = pr_auc_trapezoid(all_gt, all_pred)
    result = {
        "dataset": args.dataset_name,
        "ann_root": args.ann_root,
        "video_list": args.video_list,
        "score_folder": args.score_folder,
        "vision_folder": args.vision_folder,
        "subset_videos": len(subset),
        "videos_used": used,
        "missing_videos": missing,
        "AUC": auc,
        "AP": ap,
        "params": {
            "top_ratio": args.top_ratio,
            "temp": args.temp,
            "smooth_size": args.smooth_size,
            "smooth_sigma": args.smooth_sigma,
            "projection_mode": args.projection_mode,
            "gap_policy": args.gap_policy,
            "sampling_rate": args.sampling_rate,
            "include_self_sim": bool(args.include_self_sim),
            "apply_position_weight": bool(args.apply_position_weight),
            "position_weight_sigma": args.position_weight_sigma,
            "fuse_direct_segment_score": bool(args.fuse_direct_segment_score),
            "direct_segment_weight": args.direct_segment_weight,
        },
        "alignment_by_start_videos": aligned,
        "fallback_truncate_videos": truncated,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8-sig") as f:
            json.dump(result, f, indent=2)
    print(f"dataset={args.dataset_name}")
    print(f"subset_videos={len(subset)}")
    print(f"videos_used={used}")
    print(f"missing_videos={len(missing)}")
    print(f"AUC={auc:.9f}")
    print(f"AP={ap:.9f}")
    if args.out:
        print(f"out={args.out}")


def main():
    parser = argparse.ArgumentParser(description="Compute frame-level AUC/AP on a HIVAU subset video list.")
    parser.add_argument("--dataset-name", default="subset")
    parser.add_argument("--ann-root", required=True)
    parser.add_argument("--video-list", required=True)
    parser.add_argument("--score-folder", required=True)
    parser.add_argument("--vision-folder", required=True)
    parser.add_argument("--top-ratio", type=float, default=0.15)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--smooth-size", type=int, default=15)
    parser.add_argument("--smooth-sigma", type=float, default=10.0)
    parser.add_argument("--projection-mode", choices=["overlap", "stride"], default="overlap")
    parser.add_argument("--sampling-rate", type=int, default=-1)
    parser.add_argument("--include-self-sim", action="store_true")
    parser.add_argument("--apply-position-weight", action="store_true")
    parser.add_argument("--position-weight-sigma", type=float, default=-1.0)
    parser.add_argument("--fuse-direct-segment-score", action="store_true")
    parser.add_argument("--direct-segment-weight", type=float, default=0.50)
    parser.add_argument("--strict-alignment", action="store_true")
    parser.add_argument("--gap-policy", choices=["error", "hold", "zero"], default="error",
                        help="For stride projection: error on missing starts, hold the preceding score, or leave missing bins at zero. Initial score is zero.")
    parser.add_argument("--out", default="")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()


