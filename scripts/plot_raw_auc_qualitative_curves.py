import argparse
import json
import pathlib
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vad_detection.evaluation import compute_ucf_temporal_refinement as ucf_eval
from src.vad_detection.evaluation import compute_xd_temporal_refinement as xd_eval


DATASETS = {
    "ucf": {
        "ann_root": pathlib.Path("data/annotations/UCF_Eval.json"),
        "score_folder": pathlib.Path("results/predictions/ucf_crime_mif_stride150"),
        "vision_folder": pathlib.Path("results/cached_features/ucf_crime_stride150"),
        "videos": ["Robbery102_x264", "Fighting042_x264", "Normal_Videos_050_x264"],
        "top_ratio": 0.10,
        "temp": 10.0,
        "smooth_size": 15,
        "smooth_sigma": 10.0,
        "module": ucf_eval,
    },
    "xd": {
        "ann_root": pathlib.Path("data/annotations/xd_test_800.json"),
        "score_folder": pathlib.Path("results/predictions/xd_violence_mif_stride64"),
        "vision_folder": pathlib.Path("results/cached_features/xd_violence_stride64"),
        "videos": ["Mission.Impossible.V.Rogue.Nation.2015__#01-54-23_01-55-10_label_B2-0-0"],
        "top_ratio": 0.15,
        "temp": 10.0,
        "smooth_size": 30,
        "smooth_sigma": 30.0,
        "module": xd_eval,
    },
}


def key_from_video_path(video_path):
    name = pathlib.Path(str(video_path).replace("\\", "/")).name
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def load_annotations(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_sample(ann, video):
    for sample in ann:
        if key_from_video_path(sample["video"]) == video:
            return sample
    raise RuntimeError(f"Video not found in annotation: {video}")


def gt_ranges(sample):
    labels = sample["temporal_label"]
    ranges = []
    for i in range(0, len(labels), 2):
        start = int(labels[i])
        if start < 0:
            continue
        end = int(min(labels[i + 1], sample["length"]))
        if end > start:
            ranges.append((start, end))
    return ranges


def eval_args(cfg):
    return SimpleNamespace(
        projection_mode="stride",
        sampling_rate=-1,
        gap_policy="hold",
        top_ratio=cfg["top_ratio"],
        temp=cfg["temp"],
        smooth_size=cfg["smooth_size"],
        smooth_sigma=cfg["smooth_sigma"],
        include_self_sim=True,
        fuse_direct_segment_score=True,
        direct_segment_weight=0.1,
        apply_position_weight=True,
        position_weight_sigma=-1.0,
        use_frame_scores=False,
        frame_score_weight=0.0,
        strict_frame_score_len=False,
        strict_alignment=True,
    )


def compute_curves(dataset, video, score_folder=None):
    cfg = DATASETS[dataset]
    mod = cfg["module"]
    args = eval_args(cfg)
    ann = load_annotations(ROOT / cfg["ann_root"])
    sample = load_sample(ann, video)
    score_root = pathlib.Path(score_folder) if score_folder is not None else cfg["score_folder"]
    if not score_root.is_absolute():
        score_root = ROOT / score_root
    with (score_root / f"{video}.json").open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    features = np.load(ROOT / cfg["vision_folder"] / f"{video}.npy").astype(np.float32)

    seg_defs, seg_scores, seg_anchors = mod.parse_segments_from_payload(payload)
    seg_defs, seg_scores, seg_anchors, features, align_mode = mod.align_features_to_segment_starts(
        seg_defs,
        seg_scores,
        seg_anchors,
        features,
        payload,
        int(sample["length"]),
        strict=True,
    )

    n = len(seg_scores)
    if n == 0 or features.shape[0] != n or len(seg_defs) != n:
        raise ValueError("Expected one feature vector per saved segment.")

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms
    sim = np.dot(features, features.T)
    top_n = max(1, int(float(args.top_ratio) * n)) if n > 1 else 1
    top_n = min(top_n, n)
    top_idx = np.argsort(sim, axis=1)[:, -top_n:]
    top_values = np.take_along_axis(sim, top_idx, axis=1)

    refined_seg = []
    for i in range(n):
        weights = mod.softmax(top_values[i] * float(args.temp))
        refined_seg.append(float(np.dot(weights, seg_scores[top_idx[i]])))
    refined_seg = mod.gaussian_smooth_1d(
        np.asarray(refined_seg, dtype=np.float32),
        size=int(args.smooth_size),
        sigma=float(args.smooth_sigma),
    )

    length = int(sample["length"])
    direct = mod.project_segment_scores_to_frames(seg_scores, seg_defs, seg_anchors, payload, length, args)
    temporal = mod.project_segment_scores_to_frames(refined_seg, seg_defs, seg_anchors, payload, length, args)
    final = 0.1 * direct + 0.9 * temporal
    final = mod.gaussian_position_weight(final, sigma=float(max(1.0, length / 2.0)))

    return {
        "sample": sample,
        "direct": direct,
        "temporal": temporal,
        "final": final,
        "align_mode": align_mode,
        "n_segments": n,
    }


def axis_limit(values, fallback=1e-6):
    top = float(max(np.max(values), fallback))
    return max(top * 1.12, 0.02)


def plot_one(dataset, video, out_dir, ymax=None):
    curves = compute_curves(dataset, video)
    sample = curves["sample"]
    direct = curves["direct"]
    final = curves["final"]
    x = np.arange(len(final), dtype=np.int32)
    local_ymax = float(max(np.max(direct), np.max(final), 1e-6))
    direct_ymax = float(ymax) if ymax is not None else axis_limit(direct)
    final_ymax = float(ymax) if ymax is not None else axis_limit(final)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 5.4), dpi=160, sharex=True)
    for ax in axes:
        for start, end in gt_ranges(sample):
            ax.axvspan(start, end, color="#e15759", alpha=0.28, linewidth=0)
        ax.grid(alpha=0.25)
        ax.set_ylabel("Raw score")

    axes[0].set_ylim(0.0, direct_ymax)
    axes[1].set_ylim(0.0, final_ymax)
    axes[0].plot(x, direct, color="#4e79a7", linewidth=1.2)
    axes[0].set_title(f"{dataset.upper()} {video} - direct agent segment scores")
    axes[1].plot(x, final, color="#4e79a7", linewidth=1.2)
    axes[1].set_title("Final score used for AUC: temporal refinement + 0.1 direct fusion + position weighting")
    axes[1].set_xlabel("Frame number")

    fig.suptitle(
        f"Raw qualitative curve, no per-video min-max normalization "
        f"(segments={curves['n_segments']}, alignment={curves['align_mode']})",
        fontsize=10,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{video}_raw_auc_scores.jpg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path, local_ymax, float(np.max(final))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ucf", "xd", "both"], default="both")
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--out-dir", default="figures/qualitative/raw_auc_scores")
    parser.add_argument("--ymax", type=float, default=None)
    args = parser.parse_args()

    datasets = ["ucf", "xd"] if args.dataset == "both" else [args.dataset]
    out_dir = pathlib.Path(args.out_dir)
    written = []
    for dataset in datasets:
        videos = args.video or DATASETS[dataset]["videos"]
        for video in videos:
            out_path, local_ymax, final_max = plot_one(dataset, video, out_dir, ymax=args.ymax)
            written.append(out_path)
            print(f"wrote {out_path} raw_curve_max={local_ymax:.9f} final_auc_score_max={final_max:.9f}")

    print(f"done: {len(written)} plot(s)")


if __name__ == "__main__":
    main()
