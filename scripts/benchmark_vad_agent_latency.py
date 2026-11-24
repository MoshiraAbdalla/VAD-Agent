import argparse
import csv
import importlib
import json
import os
import pathlib
import statistics
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, GitVisionModel

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vad_detection.utils.frame_selection import sample_representative_frames_MIF
import scripts.plot_raw_auc_qualitative_curves as raw_curves

DATASETS = {
    "ucf": {
        "module": "src.vad_detection.inference.generate_ucf_mif_stride150",
        "ann_root": "data/annotations/UCF_Eval.json",
        "videos_root_env": "UCF_VIDEOS",
        "segment_stride": 150,
        "new_qs": (
            "1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior?\n"
            "2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?\n"
            "3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?\n"
            "4. Is there any visible damage or unusual movement in the video that indicates an anomaly?\n"
            "5. Are there any unusual sounds or noises in the video that suggest an anomaly?\n"
        ),
    },
    "xd": {
        "module": "src.vad_detection.inference.generate_xd_mif_stride64",
        "ann_root": "data/annotations/xd_test_800.json",
        "videos_root_env": "XD_VIDEOS",
        "segment_stride": 64,
        "new_qs": (
            "1. Are there any frames where the video content shows an object or person performing an unusual action that does not fit the usual pattern of the video content?\n"
            "2. Are there any frames where the video content shows an object or person behaving in a way that is clearly abnormal or unexpected?\n"
            "3. Are there any frames where the video content shows a clear indication of an anomaly in the behavior of objects or people, such as an object moving in an unexpected way or a person performing an unusual action?\n"
            "4. Are there any frames where the video content shows a player or object being out of place or not fitting the usual pattern of the video content?\n"
            "5. Are there any frames where the video content shows a disruption in the normal flow of the video, such as a sudden change in the camera angle or a loss of video quality?\n"
        ),
    },
}
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync_cuda()
    start = time.perf_counter()
    value = fn()
    sync_cuda()
    return value, time.perf_counter() - start


def mean(values):
    return float(statistics.fmean(values)) if values else 0.0


def stdev(values):
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    idx = int(round((len(vals) - 1) * float(pct) / 100.0))
    return vals[max(0, min(idx, len(vals) - 1))]


def token_count(tokenizer, text):
    if not text:
        return 0
    return len(tokenizer.encode(str(text), add_special_tokens=False))


def load_json(path):
    with pathlib.Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_prompts():
    with (ROOT / "configs/prompts/reasoning_agent.txt").open("r", encoding="utf-8") as f:
        reasoning = f.read()
    with (ROOT / "configs/prompts/context_agent.txt").open("r", encoding="utf-8") as f:
        context = f.read()
    with (ROOT / "configs/prompts/narrative_agent.txt").open("r", encoding="utf-8") as f:
        narrative = f.read()
    return reasoning, context, narrative


def strip_video_ext(name):
    base = pathlib.Path(str(name)).name
    lower = base.lower()
    for ext in VIDEO_EXTS:
        if lower.endswith(ext):
            return base[: -len(ext)]
    return base


def find_sample(ann_root, video_name):
    if not ann_root:
        return None
    samples = load_json(ann_root)
    samples = samples if isinstance(samples, list) else [samples]
    if video_name:
        wanted = strip_video_ext(video_name).lower()
        for sample in samples:
            if strip_video_ext(sample["video"]).lower() == wanted:
                return sample
        raise RuntimeError(f"Video {video_name!r} was not found in {ann_root}")
    return samples[0] if samples else None


def resolve_video_file(args, cfg, sample):
    if args.video_file:
        path = pathlib.Path(args.video_file)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    videos_root = pathlib.Path(args.videos_root or os.environ.get(cfg["videos_root_env"], ""))
    if not str(videos_root):
        raise RuntimeError("Pass --video-file or --videos-root for true end-to-end timing from raw video.")
    if not videos_root.exists():
        raise FileNotFoundError(videos_root)
    if sample is None:
        raise RuntimeError("Could not infer video name; pass --video-name or --video-file.")
    stem = strip_video_ext(sample["video"])
    for ext in VIDEO_EXTS:
        for candidate in (videos_root / f"{stem}{ext}", videos_root / stem / f"{stem}{ext}"):
            if candidate.exists():
                return candidate
    for ext in VIDEO_EXTS:
        matches = list(videos_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find raw video for {stem!r} under {videos_root}")


def decode_video(path, max_decode_frames=0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    meta = {
        "source_fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "source_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    frames = []
    limit = int(max_decode_frames)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if limit > 0 and len(frames) >= limit:
            break
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Decoded too few frames from {path}: {len(frames)}")
    meta["frames"] = frames
    return meta


def covered_unique_frames(intervals):
    if not intervals:
        return 0
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def stage_summary(rows, key):
    vals = [float(r[key]) for r in rows]
    return {"mean_sec": mean(vals), "std_sec": stdev(vals), "p50_sec": percentile(vals, 50), "p95_sec": percentile(vals, 95)}


def token_summary(rows, key):
    vals = [int(r[key]) for r in rows]
    return {"mean": mean(vals), "std": stdev(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95)}


def temporal_refinement_from_memory(dataset_name, length, segments, seg_features):
    if not segments or not seg_features:
        return None
    cfg = raw_curves.DATASETS[dataset_name]
    mod = cfg["module"]
    eval_args = raw_curves.eval_args(cfg)
    payload = {
        "length": int(length),
        "segment_len": int(segments[0]["end"] - segments[0]["start"]),
        "segment_stride": int(segments[1]["start"] - segments[0]["start"]) if len(segments) > 1 else int(length),
        "segments": segments,
    }
    features = np.asarray(seg_features, dtype=np.float32)
    seg_defs, seg_scores, seg_anchors = mod.parse_segments_from_payload(payload)
    n = min(len(seg_scores), features.shape[0], len(seg_defs))
    seg_defs = seg_defs[:n]
    seg_scores = seg_scores[:n]
    seg_anchors = seg_anchors[:n]
    features = features[:n]
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(features, features.T)
    top_n = max(1, int(float(eval_args.top_ratio) * n)) if n > 1 else 1
    top_n = min(top_n, n)
    top_idx = np.argsort(sim, axis=1)[:, -top_n:]
    top_values = np.take_along_axis(sim, top_idx, axis=1)
    refined_seg = []
    for i in range(n):
        weights = mod.softmax(top_values[i] * float(eval_args.temp))
        refined_seg.append(float(np.dot(weights, seg_scores[top_idx[i]])))
    refined_seg = mod.gaussian_smooth_1d(
        np.asarray(refined_seg, dtype=np.float32),
        size=int(eval_args.smooth_size),
        sigma=float(eval_args.smooth_sigma),
    )
    direct = mod.project_segment_scores_to_frames(seg_scores, seg_defs, seg_anchors, payload, int(length), eval_args)
    temporal = mod.project_segment_scores_to_frames(refined_seg, seg_defs, seg_anchors, payload, int(length), eval_args)
    final = 0.1 * direct + 0.9 * temporal
    final = mod.gaussian_position_weight(final, sigma=float(max(1.0, int(length) / 2.0)))
    return direct, temporal, final


def benchmark(args):
    cfg = DATASETS[args.dataset]
    gen = importlib.import_module(cfg["module"])
    gen.set_seed(args.seed)
    gen.make_deterministic(args.seed)
    if args.segment_stride <= 0:
        args.segment_stride = cfg["segment_stride"]

    sample = find_sample(args.ann_root or cfg["ann_root"], args.video_name)
    video_file = resolve_video_file(args, cfg, sample)
    video_name = args.video_name or strip_video_ext(video_file.name)
    print(f"Raw video: {video_file}", flush=True)
    reasoning_template, context_template, summarization_template = load_prompts()

    def load_models():
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, use_flash_attn=False, cache_dir=args.cache_dir).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, cache_dir=args.cache_dir)
        git_model = GitVisionModel.from_pretrained("microsoft/git-base", cache_dir=args.cache_dir).eval().to("cuda")
        return model, tokenizer, git_model

    (model, tokenizer, git_model), model_load_seconds = timed(load_models)
    generation_config = dict(num_beams=1, max_new_tokens=args.max_new_tokens, do_sample=False)
    image_transform = gen.build_transform(448, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    decoded, decode_seconds = timed(lambda: decode_video(video_file, max_decode_frames=args.max_decode_frames))
    frames = decoded["frames"]
    n_frms = len(frames)
    video_duration_sec = n_frms / decoded["source_fps"] if decoded["source_fps"] > 0 else 0.0
    print(f"Decoded frames={n_frms} source_fps={decoded['source_fps']:.3f} decode={decode_seconds:.3f}s", flush=True)

    rows = []
    intervals = []
    segments_for_refine = []
    features_for_refine = []
    measured_vlm_images = 0
    rolling_history_summary = "No prior context, this is the first segment of the video."
    segment_inference_start = time.perf_counter()
    max_total_segments = int(args.max_segments) if int(args.max_segments) > 0 else 10**12

    for seg_index, seg_start in enumerate(range(0, n_frms, args.segment_stride)):
        if seg_index >= max_total_segments + int(args.warmup_segments):
            break
        seg_end = min(seg_start + args.segment_len, n_frms)
        segment_start_t = time.perf_counter()
        pil_frames = frames[seg_start:seg_end]
        if len(pil_frames) < 2:
            continue

        reasoning_k = min(args.reasoning_k, len(pil_frames))
        (mif_indices_local, mif_feats), mif_seconds = timed(lambda: sample_representative_frames_MIF(pil_frames, model=git_model, K=reasoning_k, W=args.mif_w, CHUNK_SIZE=args.mif_chunk_size, return_feats=True))
        mif_indices_local = sorted(mif_indices_local)
        reason_local = mif_indices_local
        context_local = gen.uniform_indices(len(pil_frames), args.context_k)

        def prep_tensors():
            reason_pixels = [image_transform(pil_frames[i]) for i in reason_local]
            ctx_pixels = [image_transform(pil_frames[i]) for i in context_local]
            return (
                torch.stack(ctx_pixels).unsqueeze(0).to(torch.bfloat16).cuda(non_blocking=True),
                torch.stack(reason_pixels).unsqueeze(0).to(torch.bfloat16).cuda(non_blocking=True),
            )

        (ctx_pixel_values, reason_pixel_values), tensor_seconds = timed(prep_tensors)
        extra_text = "You are given a segment of a video, where the previous context in past segments is:\n" + rolling_history_summary + "\n\n The context is given for you to know what happened in the previous segments to maintain temporal dependencies through the video\n\n And, "
        ctx_question = gen.build_agent_question(context_template, len(context_local), new_qs="", extra_text="")
        reason_question = gen.build_agent_question(reasoning_template, len(reason_local), new_qs=cfg["new_qs"], extra_text=extra_text)
        ctx_input_tokens = token_count(tokenizer, ctx_question)
        reason_input_tokens = token_count(tokenizer, reason_question)

        ctx_resp, reason_resp = "", ""
        label = None
        vlm_seconds = 0.0
        for _ in range(max(1, int(args.label_retry_count) + 1)):
            (ctx_resp, reason_resp), attempt_seconds = timed(lambda: gen.batched_context_reasoning(model=model, tokenizer=tokenizer, generation_config=generation_config, ctx_pixel_values=ctx_pixel_values, reason_pixel_values=reason_pixel_values, ctx_question=ctx_question, reason_question=reason_question))
            vlm_seconds += attempt_seconds
            label = gen.extract_label_robust(reason_resp)
            if label is not None:
                break
        ctx_output_tokens = token_count(tokenizer, ctx_resp)
        reason_output_tokens = token_count(tokenizer, reason_resp)

        summary_input_tokens = 0
        summary_output_tokens = 0
        summary_seconds = 0.0
        if ctx_resp and not args.disable_summary:
            merge_text = "Previous accumulated context summary:\n" + rolling_history_summary + "\n\nNew current-segment context to merge:\n" + ctx_resp
            summary_question = summarization_template.replace("$Data", "Frame1: <image>\n") + "\n\nContext to summarize:\n" + merge_text
            summary_input_tokens = token_count(tokenizer, summary_question)
            rolling_history_summary, summary_seconds = timed(lambda: gen.summarize_with_dummy_image(model=model, tokenizer=tokenizer, generation_config=generation_config, summarization_template=summarization_template, image_transform=image_transform, text=merge_text))
            summary_output_tokens = token_count(tokenizer, rolling_history_summary)

        seg_feat, feature_seconds = timed(lambda: gen.segment_feature_from_frame_feats(mif_feats))
        segment_seconds = time.perf_counter() - segment_start_t
        warmup = seg_index < int(args.warmup_segments)
        row = {
            "dataset": args.dataset, "video": video_name, "video_file": str(video_file), "segment_index": seg_index, "warmup": int(warmup),
            "start": int(seg_start), "end": int(seg_end), "segment_frames": int(seg_end - seg_start),
            "context_frames": int(len(context_local)), "reasoning_frames": int(len(reason_local)), "vlm_images": int(len(context_local) + len(reason_local)),
            "mllm_calls_per_segment": 2 if not args.disable_summary else 1, "label": -1 if label is None else int(label),
            "mif_select_sec": mif_seconds, "tensor_prep_sec": tensor_seconds, "context_reasoning_batched_sec": vlm_seconds,
            "summary_sec": summary_seconds, "feature_sec": feature_seconds, "segment_total_pre_temporal_sec": segment_seconds,
            "context_input_tokens": ctx_input_tokens, "reasoning_input_tokens": reason_input_tokens, "summary_input_tokens": summary_input_tokens,
            "context_output_tokens": ctx_output_tokens, "reasoning_output_tokens": reason_output_tokens, "summary_output_tokens": summary_output_tokens,
        }
        rows.append(row)
        if not warmup:
            intervals.append((int(seg_start), int(seg_end)))
            measured_vlm_images += int(len(context_local) + len(reason_local) + (0 if args.disable_summary else 1))
            if label is not None and seg_feat is not None:
                segments_for_refine.append({"start": int(seg_start), "end": int(seg_end), "label": int(label)})
                features_for_refine.append(seg_feat)
        print(f"seg={seg_index} warmup={int(warmup)} frames={seg_start}-{seg_end} pre_temporal={segment_seconds:.3f}s batched_ctx_reason={vlm_seconds:.3f}s summary={summary_seconds:.3f}s label={label}", flush=True)

    segment_inference_seconds = time.perf_counter() - segment_inference_start
    temporal_refinement_seconds = 0.0
    if not args.disable_temporal_refinement:
        _, temporal_refinement_seconds = timed(lambda: temporal_refinement_from_memory(args.dataset, n_frms, segments_for_refine, features_for_refine))
    measured_rows = [r for r in rows if not r["warmup"]]
    measured_segment_seconds = sum(float(r["segment_total_pre_temporal_sec"]) for r in measured_rows)
    measured_window_frames = sum(int(r["segment_frames"]) for r in measured_rows)
    unique_frames = covered_unique_frames(intervals)
    e2e_pre_temporal = decode_seconds + segment_inference_seconds
    e2e_with_temporal = e2e_pre_temporal + temporal_refinement_seconds
    e2e_with_load = model_load_seconds + e2e_with_temporal
    lookahead_frames = max(0, n_frms - args.segment_stride) if not args.disable_temporal_refinement else 0
    lookahead_seconds = lookahead_frames / decoded["source_fps"] if decoded["source_fps"] > 0 else 0.0

    summary = {
        "dataset": args.dataset, "video": video_name, "video_file": str(video_file),
        "source_width": decoded["width"], "source_height": decoded["height"], "source_fps": decoded["source_fps"],
        "source_frame_count": decoded["source_frame_count"], "decoded_frames": n_frms, "video_duration_sec": video_duration_sec,
        "segment_len": args.segment_len, "segment_stride": args.segment_stride, "context_k": args.context_k, "reasoning_k": args.reasoning_k,
        "mllm_workload": {"calls_per_segment": 2 if not args.disable_summary else 1, "context_agent_frames": args.context_k, "reasoning_agent_frames": args.reasoning_k, "summary_agent_frames": 0 if args.disable_summary else 1, "context_and_reasoning_are_batched": True},
        "model_load_seconds": model_load_seconds, "decode_seconds": decode_seconds,
        "segment_level_inference_seconds_pre_temporal": segment_inference_seconds,
        "temporal_refinement_seconds": temporal_refinement_seconds,
        "e2e_video_seconds_excluding_model_load_pre_temporal": e2e_pre_temporal,
        "e2e_video_seconds_excluding_model_load_with_temporal": e2e_with_temporal,
        "e2e_total_seconds_including_model_load_with_temporal": e2e_with_load,
        "real_time_factor_pre_temporal_excluding_model_load": e2e_pre_temporal / video_duration_sec if video_duration_sec > 0 else 0.0,
        "real_time_factor_with_temporal_excluding_model_load": e2e_with_temporal / video_duration_sec if video_duration_sec > 0 else 0.0,
        "temporal_refinement_requires_future_segments": not args.disable_temporal_refinement,
        "additional_lookahead_delay_frames_full_video": lookahead_frames,
        "additional_lookahead_delay_seconds_full_video": lookahead_seconds,
        "warmup_segments": args.warmup_segments, "measured_segments": len(measured_rows),
        "latency_per_segment_pre_temporal": stage_summary(measured_rows, "segment_total_pre_temporal_sec"),
        "segments_per_second_pre_temporal": len(measured_rows) / measured_segment_seconds if measured_segment_seconds > 0 else 0.0,
        "window_frames_per_second_overlap_counted_pre_temporal": measured_window_frames / measured_segment_seconds if measured_segment_seconds > 0 else 0.0,
        "unique_video_frames_per_second_segments_only_pre_temporal": unique_frames / measured_segment_seconds if measured_segment_seconds > 0 else 0.0,
        "decoded_video_frames_per_second_e2e_with_temporal_excluding_model_load": n_frms / e2e_with_temporal if e2e_with_temporal > 0 else 0.0,
        "stage_seconds": {"evidence_selection_mif": stage_summary(measured_rows, "mif_select_sec"), "tensor_prep": stage_summary(measured_rows, "tensor_prep_sec"), "context_plus_reasoning_batched": stage_summary(measured_rows, "context_reasoning_batched_sec"), "narrative_memory_summary": stage_summary(measured_rows, "summary_sec"), "segment_feature": stage_summary(measured_rows, "feature_sec")},
        "token_counts": {"context_input": token_summary(measured_rows, "context_input_tokens"), "reasoning_input": token_summary(measured_rows, "reasoning_input_tokens"), "summary_input": token_summary(measured_rows, "summary_input_tokens"), "context_output": token_summary(measured_rows, "context_output_tokens"), "reasoning_output": token_summary(measured_rows, "reasoning_output_tokens"), "summary_output": token_summary(measured_rows, "summary_output_tokens")},
        "note": "Context and reasoning are timed as one batched InternVL call, matching the implemented VAD-Agent pipeline. Separate per-agent latency would require an extra diagnostic unbatched run and is not the deployed timing.",
    }

    out_dir = pathlib.Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"vad_agent_e2e_latency_{args.dataset}_{strip_video_ext(video_name)}_{stamp}"
    csv_path = out_dir / f"{base}_segments.csv"; json_path = out_dir / f"{base}_summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved_csv={csv_path}", flush=True); print(f"saved_json={json_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-end VAD-Agent latency benchmark from raw video files.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="xd")
    parser.add_argument("--video-file", default="")
    parser.add_argument("--videos-root", default="")
    parser.add_argument("--ann-root", default="")
    parser.add_argument("--video-name", default="")
    parser.add_argument("--model-path", default="OpenGVLab/InternVL2-8B")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--output-dir", default="results/latency")
    parser.add_argument("--segment-len", type=int, default=300)
    parser.add_argument("--segment-stride", type=int, default=0, help="0 means dataset default: UCF=150, XD=64.")
    parser.add_argument("--context-k", type=int, default=16)
    parser.add_argument("--reasoning-k", type=int, default=8)
    parser.add_argument("--mif-w", type=int, default=8)
    parser.add_argument("--mif-chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-segments", type=int, default=8, help="Measured segments after warmup. Use 0 for the full decoded video.")
    parser.add_argument("--warmup-segments", type=int, default=1)
    parser.add_argument("--max-decode-frames", type=int, default=0)
    parser.add_argument("--label-retry-count", type=int, default=0)
    parser.add_argument("--disable-summary", action="store_true")
    parser.add_argument("--disable-temporal-refinement", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
