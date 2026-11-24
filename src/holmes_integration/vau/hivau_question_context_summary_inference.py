#!/usr/bin/env python3
"""
Question-preserving HIVAU inference with HolmesVAU-2B plus two VAD-Agent components:

1. Context agent: produces scene/entities/context from uniformly sampled frames.
2. Summarization agent: compresses context across chunks into narrative memory.

The context agent output is used to update memory for later chunks. The answer
prompt receives the previous/accumulated summary as temporal context, matching
the agentic pipeline's "previous memory -> current segment reasoning" flow.

Unlike a binary VAD wrapper, this script keeps the original HIVAU prompt for each
instruction row. That matters because HolmesVAU was trained to answer these
question styles directly: detection/classification, description/summary,
evidence/reasoning, and clip captioning.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from holmesvau.holmesvau_utils import get_pixel_values, load_model
from holmesvau.internvl_utils import build_transform, dynamic_preprocess, get_index


DEFAULT_PROMPT_ROOT = "configs/prompts"
DEFAULT_UCF_MIF_CACHE_DIR = os.path.join(
    "agentic_inference",
    "ucf_evaluation",
    "outputs",
    "holmescls_mif160_ats_frame_scores",
)
DEFAULT_XD_MIF_CACHE_DIR = os.path.join(
    "agentic_inference",
    "xd_evaluation",
    "outputs",
    "holmescls_mif160_ats_frame_scores",
)
DEFAULT_FINAL_PROMPT_DIR = os.path.join("configs", "prompts", "holmes_question_final")
FINAL_PROMPT_FILES = {
    "judgement": "judgement.txt",
    "description": "description.txt",
    "analysis": "analysis.txt",
    "clip_caption": "clip_caption.txt",
    "default": "default.txt",
}


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_agent_prompts(prompt_root):
    root = Path(prompt_root)
    prompts = {
        "context": load_text(root / "context_agent.txt"),
        "summary": load_text(root / "narrative_agent.txt"),
    }
    evidence_path = root / "evidence_agent.txt"
    if evidence_path.exists():
        prompts["evidence"] = load_text(evidence_path)
    return prompts


def load_final_answer_prompts(prompt_dir):
    root = Path(prompt_dir)
    prompts = {}
    for key, filename in FINAL_PROMPT_FILES.items():
        path = root / filename
        if path.exists():
            prompts[key] = load_text(path)
    return prompts


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_video_index(root_path):
    if not root_path:
        return {}
    index = {}
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(".mp4") and filename not in index:
                index[filename] = os.path.join(dirpath, filename)
    return index


def resolve_video_path(hivau_root, rel_path, ucf_index=None, xd_index=None):
    if os.path.isabs(rel_path):
        return rel_path

    candidates = [
        os.path.join(hivau_root, "videos", rel_path),
        os.path.join(hivau_root, rel_path),
    ]
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate

    filename = os.path.basename(rel_path)
    if rel_path.startswith("ucf-crime") and filename in (ucf_index or {}):
        return ucf_index[filename]
    if rel_path.startswith("xd-violence") and filename in (xd_index or {}):
        return xd_index[filename]
    if filename in (ucf_index or {}):
        return ucf_index[filename]
    if filename in (xd_index or {}):
        return xd_index[filename]
    return os.path.normpath(candidates[0])


def answer_from_row(row):
    return (row.get("answer") or row.get("anwser") or "").strip()


def render_visual_prompt(template, frame_count):
    frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(frame_count))
    if "$Data" in template:
        return template.replace("$Data", frame_prefix)
    return frame_prefix + template


def chat_on_indices(model, tokenizer, generation_config, vr, frame_indices, prompt):
    if not frame_indices:
        frame_indices = [0]
    pixel_values, num_patches_list = get_pixel_values(vr, frame_indices)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    question = render_visual_prompt(prompt, len(num_patches_list))
    with torch.inference_mode():
        response, _ = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
    return response


def chat_with_dummy_image(model, tokenizer, generation_config, prompt):
    transform = build_transform(input_size=448)
    image = Image.new("RGB", (448, 448), color=(0, 0, 0))
    tiles = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=1)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    question = render_visual_prompt(prompt, 1)
    with torch.inference_mode():
        response, _ = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=[pixel_values.shape[0]],
            history=None,
            return_history=True,
        )
    return response


def llama_chat_prompt(user_prompt):
    clean_prompt = (
        str(user_prompt)
        .replace("[INST]", "")
        .replace("[/INST]", "")
        .replace("<<SYS>>", "")
        .replace("<</SYS>>", "")
    )
    return f"<s>[INST] {clean_prompt.strip()} [/INST]"


def clean_direct_summary(text):
    text = "" if text is None else str(text)
    text = text.replace("```", "").strip()
    prefixes = [
        r"^sure,\s*here\s+is\s+the\s+summary\s+of\s+the\s+full-video\s+segment\s+contexts\s*:\s*",
        r"^sure,\s*here'?s\s+the\s+summary\s+of\s+the\s+full-video\s+segment\s+contexts\s*:\s*",
        r"^sure,\s*here\s+is\s+the\s+summary\s*:\s*",
        r"^sure,\s*here'?s\s+the\s+summary\s*:\s*",
        r"^here\s+is\s+the\s+summary\s*:\s*",
        r"^here'?s\s+the\s+summary\s*:\s*",
    ]
    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    footer_markers = [
        "\nPlease note",
        "\nI have ",
        "\nI only ",
        "\nHope this helps",
        "\nLet me know",
        "\nAs an AI",
    ]
    for marker in footer_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].strip()
    return text


def dtype_from_name(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_llama_hf_summarizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.llama_model_path:
        raise ValueError("--llama_model_path is required when --summary_backend llama_hf")
    if not os.path.exists(args.llama_model_path):
        raise FileNotFoundError(f"Llama HF model path does not exist: {args.llama_model_path}")

    device = torch.device(args.summary_device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.llama_dtype)
    print(f"Loading Llama summarizer from {args.llama_model_path} on {device} ({args.llama_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_model_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    def summarize(prompt):
        text = llama_chat_prompt(prompt)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.llama_max_input_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.llama_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        return clean_direct_summary(tokenizer.decode(generated, skip_special_tokens=True))

    return summarize


def uniform_indices(vr, start, end, count):
    end = max(start + 1, min(end, len(vr)))
    count = max(1, min(int(count), end - start))
    return np.linspace(start, end - 1, num=count, dtype=int).tolist()


def ats_indices(vr, model, sampler, start, end, dense_sample_freq, select_frames):
    dense = list(range(start, end, max(1, dense_sample_freq)))
    if not dense:
        dense = [start]
    pixel_values, _ = get_pixel_values(vr, dense)
    anomaly_score, sampled = sampler.density_aware_sample(
        pixel_values,
        model,
        select_frames=min(select_frames, len(dense)),
    )
    frame_indices = [int(dense[int(i)]) for i in sampled]
    return frame_indices, np.asarray(anomaly_score, dtype=np.float32), dense


def fallback_uniform_indices(vr, count):
    idx = get_index(
        bound=None,
        fps=float(vr.get_avg_fps()),
        max_frame=len(vr) - 1,
        first_idx=0,
        num_segments=count,
    )
    return [int(i) for i in idx]


def stable_seed(base_seed, key):
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], byteorder="little", signed=False)) % (2**32)


def dataset_from_video(video_key, video_path):
    key = str(video_key).replace("\\", "/").lower()
    path = str(video_path).replace("\\", "/").lower()
    if key.startswith("ucf-crime") or "anomaly-detection-dataset" in path:
        return "ucf"
    if key.startswith("xd-violence") or "xd-violence-testing" in path:
        return "xd"
    return None


def mif_cache_path_for_video(video_key, video_path, args):
    dataset = dataset_from_video(video_key, video_path)
    if dataset == "ucf":
        cache_dir = args.ucf_mif_cache_dir
        prefix = "ucf_"
    elif dataset == "xd":
        cache_dir = args.xd_mif_cache_dir
        prefix = "xd_"
    else:
        return None, None

    stem = Path(str(video_key)).stem
    if not stem:
        stem = Path(str(video_path)).stem
    return dataset, Path(cache_dir) / f"{prefix}{stem}.json"


def load_mif_cache_for_video(video_key, video_path, args):
    dataset, cache_path = mif_cache_path_for_video(video_key, video_path, args)
    if cache_path is None or not cache_path.exists():
        return dataset, cache_path, None
    with open(cache_path, "r", encoding="utf-8-sig") as f:
        return dataset, cache_path, json.load(f)


def flatten_mif_indices(cache, n_frames):
    indices = []
    score_by_index = {}

    selected = cache.get("selected_mif_indices") or []
    scores = cache.get("selected_mif_scores") or []
    for idx, score in zip(selected, scores):
        try:
            frame_idx = int(idx)
        except Exception:
            continue
        if 0 <= frame_idx < n_frames:
            indices.append(frame_idx)
            try:
                score_by_index[frame_idx] = float(score)
            except Exception:
                pass

    for segment in cache.get("segments") or []:
        for idx in segment.get("mif_indices") or []:
            try:
                frame_idx = int(idx)
            except Exception:
                continue
            if 0 <= frame_idx < n_frames:
                indices.append(frame_idx)

    return sorted(set(indices)), score_by_index


def choose_nearest_unused(candidates, center, used):
    remaining = [idx for idx in candidates if idx not in used]
    if not remaining:
        return None
    return min(remaining, key=lambda idx: (abs(idx - center), idx))


def mif_cache_coverage_indices(vr, video_key, video_path, args, target_frames=None, mode="mif_cache_coverage"):
    n_frames = len(vr)
    if target_frames is None:
        target_frames = args.select_frames
    target = max(1, min(int(target_frames), n_frames))
    dataset, cache_path, cache = load_mif_cache_for_video(video_key, video_path, args)

    if cache is None:
        message = f"MIF cache not found for {video_key}: {cache_path}"
        if args.mif_cache_missing_policy == "uniform":
            fallback = fallback_uniform_indices(vr, target)
            return fallback, {
                "mode": mode,
                "status": "fallback_uniform",
                "reason": message,
                "dataset": dataset,
                "cache_path": None if cache_path is None else os.path.normpath(cache_path),
                "target_frames": target,
            }
        raise FileNotFoundError(message)

    mif_indices, score_by_index = flatten_mif_indices(cache, n_frames)
    if not mif_indices:
        message = f"MIF cache has no valid selected frame indices: {cache_path}"
        if args.mif_cache_missing_policy == "uniform":
            fallback = fallback_uniform_indices(vr, target)
            return fallback, {
                "mode": mode,
                "status": "fallback_uniform",
                "reason": message,
                "dataset": dataset,
                "cache_path": os.path.normpath(cache_path),
                "target_frames": target,
            }
        raise ValueError(message)

    rng = np.random.default_rng(stable_seed(args.mif_cache_seed, video_key))
    bin_edges = np.linspace(0, n_frames, target + 1, dtype=int)
    selected = []
    bin_records = []
    used = set()

    for bin_idx in range(target):
        start = int(bin_edges[bin_idx])
        end = int(bin_edges[bin_idx + 1])
        if end <= start:
            end = min(n_frames, start + 1)
        center = (start + end - 1) // 2

        candidates = [idx for idx in mif_indices if start <= idx < end and idx not in used]
        source = "bin"
        if candidates:
            chosen = int(rng.choice(candidates))
        else:
            chosen = choose_nearest_unused(mif_indices, center, used)
            source = "nearest_mif"
            if chosen is None:
                chosen = center
                source = "uniform_center"

        chosen = int(max(0, min(chosen, n_frames - 1)))
        selected.append(chosen)
        used.add(chosen)
        bin_records.append(
            {
                "bin": bin_idx,
                "start": start,
                "end": end,
                "chosen": chosen,
                "source": source,
            }
        )

    selected = sorted(set(selected))
    if len(selected) < target:
        for idx in mif_indices:
            if idx not in used:
                selected.append(int(idx))
                used.add(int(idx))
            if len(selected) >= target:
                break
    if len(selected) < target:
        for idx in fallback_uniform_indices(vr, target):
            if idx not in used:
                selected.append(int(idx))
                used.add(int(idx))
            if len(selected) >= target:
                break

    selected = sorted(selected[:target])
    selected_scores = [score_by_index.get(idx) for idx in selected]
    return selected, {
        "mode": mode,
        "status": "ok",
        "dataset": dataset,
        "cache_path": os.path.normpath(cache_path),
        "cache_length": cache.get("length"),
        "video_length": n_frames,
        "segment_len": cache.get("segment_len"),
        "segment_stride": cache.get("segment_stride"),
        "available_mif_indices": len(mif_indices),
        "target_frames": target,
        "seed": int(args.mif_cache_seed),
        "effective_seed": int(stable_seed(args.mif_cache_seed, video_key)),
        "coverage_bins": bin_records,
        "selected_mif_scores": selected_scores,
    }


def resolve_mif_uniform_counts(args, n_frames):
    if args.mif_uniform_mif_frames is None and args.mif_uniform_uniform_frames is None:
        uniform_count = max(1, int(args.select_frames) // 2)
        mif_count = max(1, int(args.select_frames) - uniform_count)
    else:
        mif_count = int(args.mif_uniform_mif_frames or 0)
        uniform_count = int(args.mif_uniform_uniform_frames or 0)
        if mif_count <= 0 and uniform_count <= 0:
            raise ValueError("MIF+uniform requires at least one selected frame from MIF or uniform.")
    mif_count = max(0, min(mif_count, n_frames))
    uniform_count = max(0, min(uniform_count, n_frames))
    return mif_count, uniform_count


def merge_named_frame_sources(source_items, n_frames, target, overlap_label):
    selected = []
    seen = set()
    source_by_frame = {}
    for source, values in source_items:
        for idx in values:
            frame_idx = int(max(0, min(int(idx), n_frames - 1)))
            if frame_idx in seen:
                source_by_frame[frame_idx] = overlap_label
                continue
            selected.append(frame_idx)
            seen.add(frame_idx)
            source_by_frame[frame_idx] = source

    if target is not None and len(selected) > target:
        selected = sorted(selected)
        keep_positions = np.linspace(0, len(selected) - 1, num=target, dtype=int).tolist()
        selected = [selected[pos] for pos in keep_positions]
    return sorted(selected), source_by_frame


def merge_frame_sources(mif_indices, uniform_indices_, n_frames, target):
    return merge_named_frame_sources(
        [("uniform", uniform_indices_), ("mif", mif_indices)],
        n_frames,
        target,
        "mif+uniform",
    )


def mif_uniform_coverage_indices(vr, video_key, video_path, args):
    n_frames = len(vr)
    mif_count, uniform_count = resolve_mif_uniform_counts(args, n_frames)
    total_target = mif_count + uniform_count
    if total_target <= 0:
        total_target = max(1, min(int(args.select_frames), n_frames))

    mif_indices = []
    mif_meta = {
        "mode": "mif_cache_coverage",
        "status": "not_requested",
        "target_frames": mif_count,
    }
    if mif_count > 0:
        mif_indices, mif_meta = mif_cache_coverage_indices(
            vr,
            video_key,
            video_path,
            args,
            target_frames=mif_count,
            mode="mif_cache_component",
        )

    uniform_indices_ = fallback_uniform_indices(vr, uniform_count) if uniform_count > 0 else []
    selected, source_by_frame = merge_frame_sources(
        mif_indices,
        uniform_indices_,
        n_frames,
        target=min(total_target, n_frames),
    )

    if len(selected) < min(total_target, n_frames):
        for idx in fallback_uniform_indices(vr, min(total_target, n_frames)):
            frame_idx = int(idx)
            if frame_idx not in source_by_frame:
                selected.append(frame_idx)
                source_by_frame[frame_idx] = "uniform_fill"
            if len(set(selected)) >= min(total_target, n_frames):
                break
    selected = sorted(set(selected))
    source_by_frame = {idx: source_by_frame.get(idx, "unknown") for idx in selected}

    return selected, {
        "mode": "mif_uniform_coverage",
        "status": "ok",
        "target_frames": len(selected),
        "mif_target_frames": mif_count,
        "uniform_target_frames": uniform_count,
        "mif_indices": [int(idx) for idx in mif_indices],
        "uniform_indices": [int(idx) for idx in uniform_indices_],
        "source_by_frame": {str(int(k)): v for k, v in sorted(source_by_frame.items())},
        "mif_component": mif_meta,
    }


def resolve_mif_ats_counts(args, n_frames):
    if args.mif_ats_mif_frames is None and args.mif_ats_ats_frames is None:
        ats_count = max(1, int(args.select_frames) // 2)
        mif_count = max(1, int(args.select_frames) - ats_count)
    else:
        mif_count = int(args.mif_ats_mif_frames or 0)
        ats_count = int(args.mif_ats_ats_frames or 0)
        if mif_count <= 0 and ats_count <= 0:
            raise ValueError("MIF+ATS requires at least one selected frame from MIF or ATS.")
    mif_count = max(0, min(mif_count, n_frames))
    ats_count = max(0, min(ats_count, n_frames))
    return mif_count, ats_count


def mif_ats_coverage_indices(vr, model, sampler, video_key, video_path, args):
    n_frames = len(vr)
    mif_count, ats_count = resolve_mif_ats_counts(args, n_frames)
    total_target = mif_count + ats_count
    if total_target <= 0:
        total_target = max(1, min(int(args.select_frames), n_frames))

    mif_indices = []
    mif_meta = {
        "mode": "mif_cache_coverage",
        "status": "not_requested",
        "target_frames": mif_count,
    }
    if mif_count > 0:
        mif_indices, mif_meta = mif_cache_coverage_indices(
            vr,
            video_key,
            video_path,
            args,
            target_frames=mif_count,
            mode="mif_cache_component",
        )

    ats_frame_indices = []
    anomaly_score = None
    dense_indices = None
    ats_meta = {
        "mode": "ats_component",
        "status": "not_requested",
        "target_frames": ats_count,
    }
    if ats_count > 0:
        try:
            ats_frame_indices, anomaly_score, dense_indices = ats_indices(
                vr,
                model,
                sampler,
                0,
                len(vr),
                dense_sample_freq=args.dense_sample_freq,
                select_frames=ats_count,
            )
            ats_meta = {
                "mode": "ats_component",
                "status": "ok",
                "dense_sample_freq": int(args.dense_sample_freq),
                "target_frames": ats_count,
            }
        except Exception as exc:
            ats_frame_indices = fallback_uniform_indices(vr, ats_count)
            anomaly_score = None
            dense_indices = None
            ats_meta = {
                "mode": "ats_component",
                "status": "fallback_uniform",
                "reason": str(exc),
                "target_frames": ats_count,
            }

    selected, source_by_frame = merge_named_frame_sources(
        [("ats", ats_frame_indices), ("mif", mif_indices)],
        n_frames,
        target=min(total_target, n_frames),
        overlap_label="mif+ats",
    )

    if len(selected) < min(total_target, n_frames):
        fill_sources = [
            ("ats_fill", fallback_uniform_indices(vr, min(total_target, n_frames))),
            ("mif_fill", mif_indices),
        ]
        for source, values in fill_sources:
            for idx in values:
                frame_idx = int(idx)
                if frame_idx not in source_by_frame:
                    selected.append(frame_idx)
                    source_by_frame[frame_idx] = source
                if len(set(selected)) >= min(total_target, n_frames):
                    break
            if len(set(selected)) >= min(total_target, n_frames):
                break
    selected = sorted(set(selected))
    source_by_frame = {idx: source_by_frame.get(idx, "unknown") for idx in selected}

    return selected, anomaly_score, dense_indices, {
        "mode": "mif_ats_coverage",
        "status": "ok",
        "target_frames": len(selected),
        "mif_target_frames": mif_count,
        "ats_target_frames": ats_count,
        "mif_indices": [int(idx) for idx in mif_indices],
        "ats_indices": [int(idx) for idx in ats_frame_indices],
        "source_by_frame": {str(int(k)): v for k, v in sorted(source_by_frame.items())},
        "mif_component": mif_meta,
        "ats_component": ats_meta,
    }


def video_chunks(num_frames, chunk_len, chunk_stride, single_chunk=False):
    if single_chunk or num_frames <= chunk_len:
        return [(0, num_frames)]
    chunks = []
    for start in range(0, num_frames, chunk_stride):
        end = min(start + chunk_len, num_frames)
        chunks.append((start, end))
        if end >= num_frames:
            break
    return chunks


def inferred_task_role(prompt, granularity):
    text = prompt.lower()
    if granularity == "clip":
        return "clip_caption"
    if any(k in text for k in ["evidence", "support", "characteristics", "judgment", "classification"]):
        return "evidence_reasoning"
    if any(k in text for k in ["summary", "description", "details", "provide about"]):
        return "description"
    if any(k in text for k in ["detect", "classify", "identify", "observed", "abnormal behaviors", "anomalies"]):
        return "detection"
    return "understanding"


def task_role(row, granularity):
    task = str(row.get("task") or "").strip()
    if task:
        return task
    return inferred_task_role(row.get("prompt") or "", granularity)


def final_question_prompt(original_prompt, granularity, previous_summary, role):
    if role in {"judgement", "detection"}:
        role_hint = (
            "The question is a detection/classification question. Answer with whether an anomaly exists "
            "and name the anomaly category when visible."
        )
    elif role == "description":
        role_hint = "The question asks for a description or summary. Describe the anomaly event and temporal order."
    elif role in {"analysis", "evidence_reasoning"}:
        role_hint = "The question asks for evidence or reasoning. Explain why the visible event is anomalous."
    elif role == "clip_caption":
        role_hint = "The question asks for a concise clip caption. Describe the visible key moments only."
    else:
        role_hint = "Answer the original question directly and concisely."

    return (
        "Use the visual frames plus the previous narrative summary below to answer the original question. "
        "Do not answer a different question.\n\n"
        f"Granularity: {granularity}\n"
        f"Question role: {role}\n"
        f"Instruction: {role_hint}\n\n"
        "Previous Narrative Summary:\n"
        f"{previous_summary}\n\n"
        "Original question:\n"
        f"{original_prompt}\n\n"
        "Answer:"
    )


def fill_final_prompt_template(template, original_prompt, granularity, previous_summary, role):
    return (
        template.replace("$Question", original_prompt)
        .replace("$Granularity", granularity)
        .replace("$Task", role)
        .replace("$Summary", previous_summary)
    )


def build_final_answer_prompt(args, prompts, original_prompt, granularity, previous_summary, role):
    if args.final_prompt_source == "original_question":
        return original_prompt

    if args.final_prompt_source == "summary_question":
        return (
            "Use the visual frames as the main evidence. Use the previous summary only as temporal context "
            "when it matches the frames.\n\n"
            "Previous summary:\n"
            f"{previous_summary}\n\n"
            "Original question:\n"
            f"{original_prompt}\n\n"
            "Answer:"
        )

    if args.final_prompt_source == "task_files":
        final_prompts = prompts.get("final") or {}
        template = final_prompts.get(role) or final_prompts.get("default")
        if template:
            return fill_final_prompt_template(template, original_prompt, granularity, previous_summary, role)

    return final_question_prompt(original_prompt, granularity, previous_summary, role)


def clip_debug_text(value, max_chars):
    text = "" if value is None else str(value)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n... [truncated]"
    return text


def print_debug_section(title, value=None, max_chars=None):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if value is not None:
        print(clip_debug_text(value, max_chars))


def print_live_debug_row(result, args):
    agent_context = result.get("agent_context") or {}
    frame_selection = result.get("frame_selection") or {}
    chunks = agent_context.get("chunks") or []
    final_prompt = agent_context.get("final_answer_prompt") or ""
    max_chars = args.print_debug_max_chars

    print_debug_section(
        (
            f"DEBUG ROW id={result.get('id')} task={result.get('task')} "
            f"source={agent_context.get('final_prompt_source')} memory={args.memory_update_mode} "
            f"summary_backend={agent_context.get('summary_backend')}"
        )
    )
    print(f"video_id: {result.get('video_id')}")
    print(f"frame_indices passed as images: {result.get('frame_indices')}")
    print(f"frame_selection.mode: {frame_selection.get('mode')}")
    print(f"source_by_frame: {frame_selection.get('source_by_frame')}")

    print_debug_section("ORIGINAL QUESTION", result.get("prompt", ""), max_chars)
    print_debug_section("SUMMARY USED IN FINAL ANSWER PROMPT", agent_context.get("summary_used_for_answer", ""), max_chars)
    print_debug_section("FINAL ANSWER MODEL TEXT INPUT", final_prompt, max_chars)
    print_debug_section("FINAL ANSWER MODEL OUTPUT / PREDICTION", result.get("pred", ""), max_chars)
    print_debug_section("REFERENCE / GT ANSWER", "\n".join(result.get("refs") or []), max_chars)

    if args.print_debug_chunks == 0:
        return

    chunk_limit = len(chunks) if args.print_debug_chunks < 0 else min(len(chunks), args.print_debug_chunks)
    print_debug_section(f"CONTEXT/SUMMARY TRACE ({len(chunks)} chunks, printing {chunk_limit})")
    for chunk in chunks[:chunk_limit]:
        print(f"\n--- chunk {chunk.get('chunk_index')} frames {chunk.get('start')}..{chunk.get('end')} ---")
        print(f"context frame indices passed as images: {chunk.get('context_indices')}")
        print("\n[Context agent output]")
        print(clip_debug_text(chunk.get("context_agent", ""), max_chars))
        print("\n[Memory before this chunk]")
        print(clip_debug_text(chunk.get("memory_before", ""), max_chars))
        if chunk.get("summary_model_prompt") is not None:
            print("\n[Exact summary model text input]")
            print(clip_debug_text(chunk.get("summary_model_prompt", ""), max_chars))
        print("\n[Memory after this chunk / final full-video memory on last chunk]")
        print(clip_debug_text(chunk.get("memory_after", ""), max_chars))

    if len(chunks) > chunk_limit:
        print(f"\n... {len(chunks) - chunk_limit} more chunks not printed. Use --print_debug_chunks -1 to show all.")


NO_PRIOR_MEMORY = "No prior context, this is the first segment of the video."


def build_summary_model_prompt(summary_template, previous_memory, context_outputs):
    merge_text = (
        "Previous accumulated context summary:\n"
        + previous_memory
        + "\n\nNew current-segment context to merge:\n"
        + "\n\n".join(context_outputs)
    )
    return summary_template + "\n\nContext to summarize:\n" + merge_text


def format_segment_context(chunk_index, start, end, context_output):
    return (
        f"Segment {int(chunk_index)} frames {int(start)}..{int(end)}:\n"
        f"{str(context_output).strip()}"
    )


def truncate_middle(text, max_chars):
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n... [middle segment contexts truncated to fit memory_max_chars] ...\n\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def build_full_context_summary_prompt(summary_template, segment_contexts, max_chars=None):
    segment_text = "\n\n".join(segment_contexts)
    segment_text = truncate_middle(segment_text, max_chars)
    return (
        summary_template
        + "\n\nTask:\n"
        + "Summarize the full-video segment contexts below into one chronological video memory. "
        + "Use the early, middle, and late segments when they contain important events. "
        + "Do not summarize only the last segment.\n\n"
        + "Full-video segment contexts:\n"
        + segment_text
    )


def summarize_memory(model, tokenizer, generation_config, summary_template, previous_memory, context_outputs, summary_text_fn=None):
    prompt = build_summary_model_prompt(summary_template, previous_memory, context_outputs)
    if summary_text_fn is not None:
        return summary_text_fn(prompt).strip()
    return chat_with_dummy_image(model, tokenizer, generation_config, prompt).strip()


def summarize_full_contexts(
    model,
    tokenizer,
    generation_config,
    summary_template,
    segment_contexts,
    max_chars=None,
    summary_text_fn=None,
):
    prompt = build_full_context_summary_prompt(summary_template, segment_contexts, max_chars=max_chars)
    if summary_text_fn is not None:
        return summary_text_fn(prompt).strip(), prompt
    return chat_with_dummy_image(model, tokenizer, generation_config, prompt).strip(), prompt


def append_context_memory(previous_memory, context_output, chunk_index, start, end, max_chars=None):
    entry = (
        f"Segment {chunk_index} frames {int(start)}..{int(end)}:\n"
        f"{str(context_output).strip()}"
    )
    if not previous_memory or previous_memory.strip() == NO_PRIOR_MEMORY:
        memory = "Accumulated video context:\n" + entry
    else:
        memory = previous_memory.rstrip() + "\n\n" + entry

    if max_chars and max_chars > 0 and len(memory) > max_chars:
        prefix = "Accumulated video context (earliest details truncated to fit):\n"
        memory = prefix + memory[-max_chars:]
    return memory


def build_context_memory_trace(
    model,
    tokenizer,
    generation_config,
    vr,
    prompts,
    context_frames,
    chunk_len,
    chunk_stride,
    single_chunk,
    memory_update_mode="model",
    memory_max_chars=None,
    store_summary_prompt=False,
    summary_text_fn=None,
):
    chunks = video_chunks(len(vr), chunk_len, chunk_stride, single_chunk=single_chunk)
    memory = NO_PRIOR_MEMORY
    chunk_records = []
    segment_contexts = []

    for idx, (start, end) in enumerate(chunks):
        memory_before = memory
        frame_indices = uniform_indices(vr, start, end, context_frames)
        context_output = chat_on_indices(
            model,
            tokenizer,
            generation_config,
            vr,
            frame_indices,
            prompts["context"],
        ).strip()
        summary_model_prompt = None
        segment_context = format_segment_context(idx, start, end, context_output)
        segment_contexts.append(segment_context)
        if memory_update_mode in {"append_context", "full_context_summary"}:
            memory = append_context_memory(
                memory,
                context_output,
                idx,
                start,
                end,
                max_chars=memory_max_chars,
            )
        else:
            summary_model_prompt = build_summary_model_prompt(prompts["summary"], memory, [context_output])
            memory = summarize_memory(
                model,
                tokenizer,
                generation_config,
                prompts["summary"],
                memory,
                [context_output],
                summary_text_fn=summary_text_fn,
            ) or memory
        chunk_records.append(
            {
                "chunk_index": idx,
                "start": int(start),
                "end": int(end),
                "context_indices": frame_indices,
                "memory_before": memory_before,
                "context_agent": context_output,
                "memory_after": memory,
            }
        )
        if store_summary_prompt:
            chunk_records[-1]["summary_model_prompt"] = summary_model_prompt

    if memory_update_mode == "full_context_summary" and chunk_records:
        memory, summary_model_prompt = summarize_full_contexts(
            model,
            tokenizer,
            generation_config,
            prompts["summary"],
            segment_contexts,
            max_chars=memory_max_chars,
            summary_text_fn=summary_text_fn,
        )
        chunk_records[-1]["memory_after_context_collection"] = chunk_records[-1]["memory_after"]
        chunk_records[-1]["memory_after"] = memory
        if store_summary_prompt:
            chunk_records[-1]["summary_model_prompt"] = summary_model_prompt

    return memory, chunk_records


def previous_summary_for_answer(chunk_records):
    if not chunk_records:
        return "No prior context, this is the first segment of the video."
    # For whole-video or event/clip answering after scanning the segment, the
    # useful previous summary is the final accumulated memory. During true
    # per-segment reasoning, use each chunk's memory_before from chunk_records.
    return chunk_records[-1]["memory_after"]


def expanded_summary_indices(final_indices, num_frames, radius=0, max_frames=0):
    values = []
    radius = int(radius or 0)
    for idx in final_indices:
        frame_idx = int(idx)
        if radius > 0:
            values.extend([frame_idx - radius, frame_idx, frame_idx + radius])
        else:
            values.append(frame_idx)
    values = sorted({max(0, min(num_frames - 1, int(idx))) for idx in values})
    if max_frames and max_frames > 0 and len(values) > max_frames:
        keep = np.linspace(0, len(values) - 1, num=max_frames, dtype=int).tolist()
        values = [values[i] for i in keep]
    return values


def build_final_frame_evidence_summary(model, tokenizer, generation_config, vr, prompts, final_indices, args):
    summary_indices = expanded_summary_indices(
        final_indices,
        len(vr),
        radius=args.summary_final_frame_neighbor_radius,
        max_frames=args.summary_final_frame_max_frames,
    )
    prompt = prompts.get("evidence") or (
        "Describe the anomaly-relevant visual evidence in these selected frames. "
        "Use only visible facts. Keep chronological order. Do not invent a category.\n\n"
        "[$Data]\n"
        "Summary:"
    )
    summary = chat_on_indices(model, tokenizer, generation_config, vr, summary_indices, prompt).strip()
    return summary, prompt, summary_indices


def plain_list(values):
    if values is None:
        return None
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def run_question_row(
    row,
    granularity,
    video_path,
    model,
    tokenizer,
    generation_config,
    sampler,
    prompts,
    args,
    cached_context=None,
    cached_final_frames=None,
    final_frame_cache=None,
    summary_text_fn=None,
):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    if args.holmes_only:
        memory_text = "Holmes-only baseline: no context or summarization agent was used."
        chunk_records = []
    elif args.summary_from_final_frames and cached_context is None:
        memory_text = "Final-frame evidence summary will be generated after final frame selection."
        chunk_records = []
    elif cached_context is None:
        memory_text, chunk_records = build_context_memory_trace(
            model,
            tokenizer,
            generation_config,
            vr,
            prompts,
            context_frames=args.context_frames,
            chunk_len=args.summary_chunk_len,
            chunk_stride=args.summary_chunk_stride,
            single_chunk=args.single_chunk or granularity == "clip",
            memory_update_mode=args.memory_update_mode,
            memory_max_chars=args.memory_max_chars,
            store_summary_prompt=args.store_summary_prompt,
            summary_text_fn=summary_text_fn,
        )
    else:
        memory_text, chunk_records = cached_context

    frame_cache_key = row["video"]
    if cached_final_frames is None and final_frame_cache is not None:
        cached_final_frames = final_frame_cache.get(frame_cache_key)
    if cached_final_frames is not None:
        final_indices = list(cached_final_frames["frame_indices"])
        dense_indices = cached_final_frames.get("dense_indices")
        anomaly_score = cached_final_frames.get("anomaly_score")
        frame_selection = cached_final_frames["frame_selection"]
    else:
        frame_selection = {"mode": "uniform", "status": "ok"}
        if args.use_MIF_ATS_frames:
            final_indices, anomaly_score, dense_indices, frame_selection = mif_ats_coverage_indices(
                vr,
                model,
                sampler,
                row["video"],
                video_path,
                args,
            )
        elif args.use_MIF_uniform_frames:
            final_indices, frame_selection = mif_uniform_coverage_indices(
                vr,
                row["video"],
                video_path,
                args,
            )
            anomaly_score = None
            dense_indices = None
        elif args.use_MIF_cache_frames:
            final_indices, frame_selection = mif_cache_coverage_indices(
                vr,
                row["video"],
                video_path,
                args,
            )
            anomaly_score = None
            dense_indices = None
        elif args.use_ATS:
            try:
                final_indices, anomaly_score, dense_indices = ats_indices(
                    vr,
                    model,
                    sampler,
                    0,
                    len(vr),
                    dense_sample_freq=args.dense_sample_freq,
                    select_frames=args.select_frames,
                )
                frame_selection = {
                    "mode": "ats",
                    "status": "ok",
                    "dense_sample_freq": int(args.dense_sample_freq),
                    "target_frames": int(args.select_frames),
                }
            except Exception:
                final_indices = fallback_uniform_indices(vr, args.select_frames)
                anomaly_score = None
                dense_indices = None
                frame_selection = {
                    "mode": "ats",
                    "status": "fallback_uniform",
                    "target_frames": int(args.select_frames),
                }
        else:
            final_indices = fallback_uniform_indices(vr, args.select_frames)
            anomaly_score = None
            dense_indices = None
            frame_selection = {
                "mode": "uniform",
                "status": "ok",
                "target_frames": int(args.select_frames),
            }

        if final_frame_cache is not None:
            final_frame_cache[frame_cache_key] = {
                "frame_indices": [int(idx) for idx in final_indices],
                "dense_indices": plain_list(dense_indices),
                "anomaly_score": plain_list(anomaly_score),
                "frame_selection": frame_selection,
            }

    if args.summary_from_final_frames and not args.holmes_only and cached_context is None:
        memory_text, evidence_prompt, summary_indices = build_final_frame_evidence_summary(
            model,
            tokenizer,
            generation_config,
            vr,
            prompts,
            final_indices,
            args,
        )
        chunk_records = [
            {
                "chunk_index": 0,
                "start": 0,
                "end": int(len(vr)),
                "context_indices": [int(idx) for idx in summary_indices],
                "answer_frame_indices": [int(idx) for idx in final_indices],
                "memory_before": NO_PRIOR_MEMORY,
                "context_agent": memory_text,
                "memory_after": memory_text,
                "summary_type": "final_frame_evidence",
            }
        ]
        if args.store_summary_prompt:
            chunk_records[-1]["summary_model_prompt"] = evidence_prompt

    original_prompt = row.get("prompt") or ""
    dataset_task = task_role(row, granularity)
    prompt_inferred_task = inferred_task_role(original_prompt, granularity)
    if args.holmes_only:
        answer_prompt = original_prompt
        previous_summary = memory_text
    else:
        previous_summary = previous_summary_for_answer(chunk_records)
        answer_prompt = build_final_answer_prompt(
            args,
            prompts,
            original_prompt,
            granularity,
            previous_summary,
            dataset_task,
        )
    pred = chat_on_indices(model, tokenizer, generation_config, vr, final_indices, answer_prompt).strip()

    return {
        "id": str(row.get("id")),
        "video_id": row["video"],
        "video_name": os.path.basename(row["video"]),
        "video_path": video_path,
        "granularity": granularity,
        "prompt": original_prompt,
        "task_role": dataset_task,
        "task": dataset_task,
        "inferred_task_role": prompt_inferred_task,
        "refs": [answer_from_row(row)],
        "pred": pred,
        "frame_indices": final_indices,
        "dense_indices": dense_indices,
        "anomaly_score": plain_list(anomaly_score),
        "frame_selection": frame_selection,
        "agent_context": {
            "mode": "holmes_only" if args.holmes_only else "context_summary",
            "summary_backend": "none" if args.holmes_only else args.summary_backend,
            "summary_model_path": None if args.holmes_only or args.summary_backend == "holmes" else args.llama_model_path,
            "final_prompt_source": "original_question" if args.holmes_only else args.final_prompt_source,
            "final_answer_prompt_dir": None if args.holmes_only else os.path.normpath(args.final_answer_prompt_dir),
            "reuse_final_frames_from": None if args.holmes_only else args.reuse_final_frames_from,
            "summary_used_for_answer": previous_summary,
            "final_summary": memory_text,
            "final_answer_prompt": answer_prompt,
            "chunks": chunk_records,
        },
    }, (memory_text, chunk_records)


def save_json_atomic(path, payload):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)


def context_tuple_from_result(item):
    agent_context = item.get("agent_context")
    if not isinstance(agent_context, dict):
        return None
    final_summary = agent_context.get("final_summary")
    chunks = agent_context.get("chunks")
    if final_summary is None or chunks is None:
        return None
    return str(final_summary), chunks


def load_reused_agent_contexts(path):
    if not path:
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    by_id = {}
    by_video = {}
    for item in rows:
        context = context_tuple_from_result(item)
        if context is None:
            continue
        row_id = item.get("id")
        if row_id is not None:
            by_id[str(row_id)] = context
        video_id = item.get("video_id")
        if video_id and video_id not in by_video:
            by_video[str(video_id)] = context
    return by_id, by_video


def final_frames_from_result(item):
    frame_indices = item.get("frame_indices")
    if not frame_indices:
        return None
    return {
        "frame_indices": [int(idx) for idx in frame_indices],
        "dense_indices": item.get("dense_indices"),
        "anomaly_score": item.get("anomaly_score"),
        "frame_selection": item.get("frame_selection") or {"mode": "reused", "status": "ok"},
    }


def load_reused_final_frames(path):
    if not path:
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    by_id = {}
    by_video = {}
    for item in rows:
        final_frames = final_frames_from_result(item)
        if final_frames is None:
            continue
        row_id = item.get("id")
        if row_id is not None:
            by_id[str(row_id)] = final_frames
        video_id = item.get("video_id")
        if video_id and video_id not in by_video:
            by_video[str(video_id)] = final_frames
    return by_id, by_video


def main():
    parser = argparse.ArgumentParser(description="Question-preserving video inference with VAD-Agent context + narrative agents.")
    parser.add_argument("--ckpt", default="HolmesVAU-2B")
    parser.add_argument("--sampler_path", default=os.path.join("holmesvau", "ATS", "anomaly_scorer.pth"))
    parser.add_argument("--hivau_root", default="HIVAU-70k")
    parser.add_argument("--prompt_root", default=DEFAULT_PROMPT_ROOT)
    parser.add_argument("--final_answer_prompt_dir", default=DEFAULT_FINAL_PROMPT_DIR)
    parser.add_argument(
        "--final_prompt_source",
        choices=["legacy", "task_files", "original_question", "summary_question"],
        default="legacy",
        help=(
            "legacy keeps the old hard-coded wrapper; task_files uses task-specific .txt prompts; "
            "original_question asks Holmes with the original prompt only; "
            "summary_question uses a minimal summary plus original-question prompt."
        ),
    )
    parser.add_argument("--ucf_root")
    parser.add_argument("--xd_root")
    parser.add_argument("--granularity", choices=["video", "event", "clip"], default="video")
    parser.add_argument("--out", default=os.path.join("agentic_inference", "outputs", "hivau_video_question_context_summary_predictions.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--select_frames", type=int, default=12)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--summary_chunk_len", type=int, default=480)
    parser.add_argument("--summary_chunk_stride", type=int, default=480)
    parser.add_argument(
        "--memory_update_mode",
        choices=["model", "append_context", "full_context_summary"],
        default="model",
        help=(
            "model keeps the original iterative summary-agent behavior; append_context builds a deterministic "
            "running memory from every segment context; full_context_summary collects all segment contexts "
            "and calls the summary model once after seeing the whole video."
        ),
    )
    parser.add_argument(
        "--memory_max_chars",
        type=int,
        default=6000,
        help="Maximum characters for append_context running memory. Use <=0 for no truncation.",
    )
    parser.add_argument(
        "--store_summary_prompt",
        action="store_true",
        help="Store the exact summary model prompt in chunk records for debugging.",
    )
    parser.add_argument(
        "--summary_from_final_frames",
        action="store_true",
        help="Build the summary from the same final frames passed to the answer model, instead of chunk memory.",
    )
    parser.add_argument(
        "--summary_final_frame_neighbor_radius",
        type=int,
        default=0,
        help="When --summary_from_final_frames is set, also include +/- this many frames around each final frame in the summary-only visual input.",
    )
    parser.add_argument(
        "--summary_final_frame_max_frames",
        type=int,
        default=0,
        help="Maximum summary-only frames after neighbor expansion. Use <=0 for no cap.",
    )
    parser.add_argument(
        "--summary_backend",
        choices=["holmes", "llama_hf"],
        default="holmes",
        help="Text model used for summary updates. holmes keeps the original dummy-image HolmesVAU summarizer.",
    )
    parser.add_argument(
        "--llama_model_path",
        default=os.path.join("agentic_inference", "models", "Llama-2-13b-chat-hf"),
        help="Converted HuggingFace Llama model path used when --summary_backend llama_hf.",
    )
    parser.add_argument(
        "--summary_device",
        default="cuda:0",
        help="Device for the Llama text summarizer when --summary_backend llama_hf.",
    )
    parser.add_argument(
        "--llama_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype for loading the Llama text summarizer.",
    )
    parser.add_argument(
        "--llama_max_input_tokens",
        type=int,
        default=3584,
        help="Maximum tokenized prompt length passed to Llama summary generation.",
    )
    parser.add_argument(
        "--llama_max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for each Llama summary.",
    )
    parser.add_argument(
        "--print_debug_io",
        action="store_true",
        help="Print selected model inputs/outputs during inference for debugging.",
    )
    parser.add_argument(
        "--print_debug_limit",
        type=int,
        default=1,
        help="Maximum number of newly processed rows to print when --print_debug_io is set. Use 0 for all.",
    )
    parser.add_argument(
        "--print_debug_max_chars",
        type=int,
        default=2500,
        help="Maximum characters per debug text block. Use <=0 for no truncation.",
    )
    parser.add_argument(
        "--print_debug_chunks",
        type=int,
        default=3,
        help="Number of context chunks to print per debug row. Use -1 for all chunks, 0 for none.",
    )
    parser.add_argument("--single_chunk", action="store_true")
    parser.add_argument("--dense_sample_freq", type=int, default=16)
    parser.add_argument("--use_ATS", action="store_true")
    parser.add_argument(
        "--use_MIF_cache_frames",
        action="store_true",
        help="Use saved full-dataset MIF-selected frame caches for the final answer frames instead of ATS.",
    )
    parser.add_argument(
        "--use_MIF_uniform_frames",
        action="store_true",
        help="Use a MIF-cache plus uniform coverage mix for the final answer frames instead of ATS.",
    )
    parser.add_argument(
        "--use_MIF_ATS_frames",
        action="store_true",
        help="Use a MIF-cache plus ATS mix for the final answer frames.",
    )
    parser.add_argument(
        "--mif_uniform_mif_frames",
        type=int,
        default=None,
        help="Number of final answer frames to take from saved MIF cache. Default splits --select_frames with uniform.",
    )
    parser.add_argument(
        "--mif_uniform_uniform_frames",
        type=int,
        default=None,
        help="Number of final answer frames to take from uniform coverage. Default splits --select_frames with MIF.",
    )
    parser.add_argument(
        "--mif_ats_mif_frames",
        type=int,
        default=None,
        help="Number of final answer frames to take from saved MIF cache. Default splits --select_frames with ATS.",
    )
    parser.add_argument(
        "--mif_ats_ats_frames",
        type=int,
        default=None,
        help="Number of final answer frames to take from ATS. Default splits --select_frames with MIF.",
    )
    parser.add_argument(
        "--reuse_agent_context_from",
        default=None,
        help="Load agent_context summaries from an existing question-level output and rerun only final answers.",
    )
    parser.add_argument(
        "--reuse_final_frames_from",
        default=None,
        help="Load final frame indices from an existing question-level output and rerun only final answers/prompts.",
    )
    parser.add_argument("--ucf_mif_cache_dir", default=DEFAULT_UCF_MIF_CACHE_DIR)
    parser.add_argument("--xd_mif_cache_dir", default=DEFAULT_XD_MIF_CACHE_DIR)
    parser.add_argument("--mif_cache_seed", type=int, default=2026)
    parser.add_argument(
        "--mif_cache_missing_policy",
        choices=["error", "uniform"],
        default="error",
        help="What to do if a saved MIF cache is missing for a video.",
    )
    parser.add_argument("--holmes_only", action="store_true", help="Question-level Holmes baseline: answer original prompts without context/summarization.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    if args.print_debug_io:
        args.store_summary_prompt = True

    final_frame_modes = [
        args.use_ATS,
        args.use_MIF_cache_frames,
        args.use_MIF_uniform_frames,
        args.use_MIF_ATS_frames,
    ]
    if sum(bool(mode) for mode in final_frame_modes) > 1:
        parser.error("Choose only one final-frame mode: --use_ATS, --use_MIF_cache_frames, --use_MIF_uniform_frames, or --use_MIF_ATS_frames.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prompts = load_agent_prompts(args.prompt_root)
    prompts["final"] = load_final_answer_prompts(args.final_answer_prompt_dir)
    print(f"Loading HolmesVAU-2B from {args.ckpt} on {device}")
    model, tokenizer, generation_config, sampler = load_model(args.ckpt, args.sampler_path, device)
    generation_config["max_new_tokens"] = args.max_new_tokens
    generation_config["do_sample"] = False

    rows = load_jsonl(os.path.join(args.hivau_root, "instruction", "merge_instruction_test_final.jsonl"))
    rows = [r for r in rows if r.get("type") == args.granularity and answer_from_row(r)]
    task_by_id = {
        str(row.get("id")): str(row.get("task") or "").strip()
        for row in rows
        if row.get("id") is not None and str(row.get("task") or "").strip()
    }

    if args.max_videos is not None:
        allowed = []
        seen = set()
        for row in rows:
            if row["video"] not in seen:
                if len(seen) >= args.max_videos:
                    break
                seen.add(row["video"])
            allowed.append(row)
        rows = allowed
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    ucf_index = build_video_index(args.ucf_root)
    xd_index = build_video_index(args.xd_root)

    results = []
    done_ids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            results = json.load(f)
        for item in results:
            dataset_task = task_by_id.get(str(item.get("id")))
            if not dataset_task:
                continue
            old_role = str(item.get("task_role") or "").strip()
            if old_role and old_role != dataset_task:
                item.setdefault("inferred_task_role", old_role)
            item["task_role"] = dataset_task
            item["task"] = dataset_task
        done_ids = {str(item.get("id")) for item in results}
        print(f"Resuming from {args.out}; loaded {len(done_ids)} completed rows")

    reused_context_by_id, reused_context_by_video = load_reused_agent_contexts(args.reuse_agent_context_from)
    if args.reuse_agent_context_from:
        print(
            f"Loaded reusable agent contexts from {args.reuse_agent_context_from}: "
            f"{len(reused_context_by_id)} by id, {len(reused_context_by_video)} by video"
        )
    reused_final_frames_by_id, reused_final_frames_by_video = load_reused_final_frames(args.reuse_final_frames_from)
    if args.reuse_final_frames_from:
        print(
            f"Loaded reusable final frames from {args.reuse_final_frames_from}: "
            f"{len(reused_final_frames_by_id)} by id, {len(reused_final_frames_by_video)} by video"
        )

    context_cache = {}
    final_frame_cache = {}
    new_count = 0
    debug_printed = 0
    pending_rows = [row for row in rows if str(row.get("id")) not in done_ids]
    summary_text_fn = None
    if args.summary_backend == "llama_hf" and not args.holmes_only and pending_rows:
        summary_text_fn = load_llama_hf_summarizer(args)

    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id"))
        if row_id in done_ids:
            continue
        video_path = resolve_video_path(args.hivau_root, row["video"], ucf_index=ucf_index, xd_index=xd_index)
        if not os.path.exists(video_path):
            print(f"[{idx}/{len(rows)}] missing {video_path}")
            continue

        print(
            f"[{idx}/{len(rows)}] {args.granularity} id={row_id} video={row['video']} "
            f"task={task_role(row, args.granularity)} inferred={inferred_task_role(row.get('prompt', ''), args.granularity)}"
        )
        cached = (
            reused_context_by_id.get(row_id)
            or reused_context_by_video.get(row["video"])
            or context_cache.get(row["video"])
        )
        cached_final_frames = (
            reused_final_frames_by_id.get(row_id)
            or reused_final_frames_by_video.get(row["video"])
            or final_frame_cache.get(row["video"])
        )
        try:
            result, cached = run_question_row(
                row,
                args.granularity,
                video_path,
                model,
                tokenizer,
                generation_config,
                sampler,
                prompts,
                args,
                cached_context=cached,
                cached_final_frames=cached_final_frames,
                final_frame_cache=final_frame_cache,
                summary_text_fn=summary_text_fn,
            )
        except Exception as exc:
            print(f"  failed id={row_id}: {exc}")
            continue

        context_cache[row["video"]] = cached
        results.append(result)
        done_ids.add(row_id)
        new_count += 1
        if args.print_debug_io and (args.print_debug_limit <= 0 or debug_printed < args.print_debug_limit):
            print_live_debug_row(result, args)
            debug_printed += 1
        if args.save_every > 0 and new_count % args.save_every == 0:
            save_json_atomic(args.out, results)
            print(f"  saved {args.out}")

    save_json_atomic(args.out, results)
    print(f"Saved {len(results)} question-level predictions to {args.out}")


if __name__ == "__main__":
    main()




