import argparse
import json
import os
import pathlib
import random
import re
import time

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def make_deterministic(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EnsureRGB:
    def __call__(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_transform(input_size, mean, std):
    return T.Compose(
        [
            EnsureRGB(),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class VideoInstructDatasetInference(Dataset):
    def __init__(self, vis_root, ann_root):
        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding="utf-8") as f:
            self.annotation = json.load(f)

        self.vis_root = vis_root
        self.transform = build_transform(
            448,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        sample = self.annotation[index]
        video_name = sample["video"].split("/")[-1].split(".")[0]
        video_path = os.path.join(self.vis_root, video_name)
        vlen = sample["length"]
        return video_path, video_name, vlen


def build_agent_question(model_instruct, frame_count, new_qs="", extra_text=""):
    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(frame_count)])
    question = model_instruct.replace("$Data", video_prefix)
    position_flag = (
        "Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' "
        "in 'Yes, there is an anomaly' or 'No, there is no anomaly'."
    )
    q_start_index = question.find(position_flag)
    if q_start_index != -1 and new_qs:
        question = question[:q_start_index] + new_qs + question[q_start_index:]
    if extra_text:
        question = extra_text + "\n\n" + question
    return question


def summarize_with_dummy_image(model, tokenizer, generation_config, summarization_template, image_transform, text):
    dummy = Image.new("RGB", (448, 448), color=(0, 0, 0))
    pixel_values = image_transform(dummy).unsqueeze(0).to(torch.bfloat16).cuda(non_blocking=True)
    question = summarization_template.replace("$Data", "Frame1: <image>\n")
    question = question + "\n\nContext to summarize:\n" + text
    with torch.inference_mode():
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=[1],
            questions=[question],
            generation_config=generation_config,
        )
    return responses[0] if responses else ""


def batched_context_reasoning(
    model,
    tokenizer,
    generation_config,
    ctx_pixel_values,
    reason_pixel_values,
    ctx_question,
    reason_question,
):
    ctx_frame_size = int(ctx_pixel_values.size(1))
    reason_frame_size = int(reason_pixel_values.size(1))

    ctx_flat = torch.reshape(
        ctx_pixel_values.to(torch.bfloat16),
        (ctx_pixel_values.size(0) * ctx_frame_size, ctx_pixel_values.size(2), ctx_pixel_values.size(3), ctx_pixel_values.size(4)),
    )
    reason_flat = torch.reshape(
        reason_pixel_values.to(torch.bfloat16),
        (reason_pixel_values.size(0) * reason_frame_size, reason_pixel_values.size(2), reason_pixel_values.size(3), reason_pixel_values.size(4)),
    )

    all_pixels = torch.cat([ctx_flat, reason_flat], dim=0)
    with torch.inference_mode():
        responses = model.batch_chat(
            tokenizer,
            all_pixels,
            num_patches_list=[ctx_frame_size, reason_frame_size],
            questions=[ctx_question, reason_question],
            generation_config=generation_config,
        )

    ctx_response = responses[0] if len(responses) > 0 else ""
    reason_response = responses[1] if len(responses) > 1 else ""
    return ctx_response, reason_response


def extract_label_robust(response: str):
    if not response:
        return None
    text = response.strip()
    m = re.search(r"(?is)\bOutput\b\s*:\s*(?:\[\s*)?([01])(?:\s*\])?", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?m)^\s*([01])\s*$", text)
    if m:
        return int(m.group(1))
    return None


def normalize_frame_scores(scores, length):
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), mode="constant")
    elif arr.size > length:
        arr = arr[:length]
    return arr.astype(np.float32)


def load_segment_frames_with_global_ids(video_dir, start, end, fps_downsample=1):
    """Load frames in [start, end) and return aligned (frames, global_ids)."""
    start_i = int(start)
    end_i = int(end)
    step = int(max(1, fps_downsample))
    frames = []
    global_ids = []
    for fid in range(start_i, end_i, step):
        fp = os.path.join(video_dir, f"{fid:06d}.jpg")
        if not os.path.exists(fp):
            continue
        with Image.open(fp) as im:
            frames.append(im.convert("RGB").copy())
        global_ids.append(int(fid))
    return frames, global_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-root", default="data/ucf/frames/")
    parser.add_argument("--ann-root", default="data/annotations/UCF_Eval.json")
    parser.add_argument("--source-score-folder", default="results/predictions/ucf_crime_mif_stride150/")
    parser.add_argument("--save-folder", default="results/predictions/ucf_ablation_mif8_mif8/")
    parser.add_argument("--segment-len", type=int, default=300)
    parser.add_argument("--segment-stride", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-videos", type=int, default=290)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-model-output", action="store_true")
    parser.add_argument(
        "--label-retry-count",
        type=int,
        default=1,
        help="Number of extra retries for the same segment when label parsing fails.",
    )
    parser.add_argument(
        "--label-retry-wait-sec",
        type=float,
        default=1.0,
        help="Seconds to wait before retrying an unparsable segment response.",
    )
    args = parser.parse_args()

    set_seed(42)
    make_deterministic(42)
    os.makedirs(args.save_folder, exist_ok=True)
    log("deterministic mode enabled; starting UCF MIF-only full pipeline (reuse stride-150 MIF indices) run")

    with open("configs/prompts/reasoning_agent.txt", "r", encoding="utf-8") as f:
        model_instruct = f.read()
    with open("configs/prompts/context_agent.txt", "r", encoding="utf-8") as f:
        model_context_instruct = f.read()
    with open("configs/prompts/narrative_agent.txt", "r", encoding="utf-8") as f:
        summarization_template = f.read()

    new_qs = (
        "1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior?\n"
        "2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?\n"
        "3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?\n"
        "4. Is there any visible damage or unusual movement in the video that indicates an anomaly?\n"
        "5. Are there any unusual sounds or noises in the video that suggest an anomaly?\n"
    )

    path = "OpenGVLab/InternVL2-8B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        cache_dir="./cache",
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir="./cache")
    generation_config = dict(num_beams=1, max_new_tokens=args.max_new_tokens, do_sample=False)

    dataset = VideoInstructDatasetInference(vis_root=args.vis_root, ann_root=args.ann_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    image_transform = dataset.transform

    target_total = min(len(dataset), int(args.max_videos))
    existing_done = 0
    for idx in range(target_total):
        sample = dataset.annotation[idx]
        vn = sample["video"].split("/")[-1].split(".")[0]
        out_json_i = os.path.join(args.save_folder, f"{vn}.json")
        if os.path.exists(out_json_i):
            existing_done += 1
    log(f"resume status: already_done={existing_done}/{target_total}")

    done = 0
    for video_path, video_name, n_frms in loader:
        video_path = video_path[0]
        video_name = video_name[0]
        n_frms = int(n_frms.item())

        out_json = os.path.join(args.save_folder, f"{video_name}.json")
        if os.path.exists(out_json):
            continue

        source_json = os.path.join(args.source_score_folder, f"{video_name}.json")
        if not os.path.exists(source_json):
            log(f"skip video={video_name} reason=missing_source_json path={source_json}")
            continue

        log(f"processing video={video_name} frames={n_frms}")
        rolling_history_summary = "No prior context, this is the first segment of the video."
        segments = []

        with open(source_json, "r", encoding="utf-8") as f:
            src_payload = json.load(f)
        src_segments = src_payload.get("segments", [])
        src_frame_scores = normalize_frame_scores(src_payload.get("frame_scores", []), n_frms)

        for src_seg in src_segments:
            seg_start = int(src_seg.get("start", 0))
            seg_end = int(src_seg.get("end", seg_start))
            if seg_end <= seg_start:
                log(f"skip segment [{seg_start},{seg_end}) reason=invalid_bounds")
                continue

            pil_frames, frame_ids = load_segment_frames_with_global_ids(
                video_path,
                seg_start,
                seg_end,
                fps_downsample=1,
            )
            if len(pil_frames) < 2:
                log(f"skip segment [{seg_start},{seg_end}) reason=too_few_frames")
                continue

            id_to_local = {int(fid): idx for idx, fid in enumerate(frame_ids)}
            reason_global_source = [int(x) for x in src_seg.get("mif_reasoning_indices", [])]
            reason_global_in_window = [g for g in reason_global_source if seg_start <= g < seg_end]
            dropped_missing = [g for g in reason_global_in_window if g not in id_to_local]
            reason_local = [id_to_local[g] for g in reason_global_source if g in id_to_local]
            reason_local = list(dict.fromkeys(reason_local))
            if len(reason_local) == 0:
                log(
                    f"skip segment [{seg_start},{seg_end}) reason=missing_precomputed_mif_indices "
                    f"source_global={reason_global_source} in_window={reason_global_in_window} "
                    f"dropped_missing={dropped_missing} loaded_global_count={len(frame_ids)}"
                )
                continue

            # MIF-only: feed the same most-anomalous frames to both context and reasoning agents.
            context_local = reason_local
            context_global = [int(frame_ids[i]) for i in context_local]
            reason_global_used = [int(frame_ids[i]) for i in reason_local]
            log(
                f"sampled_frames segment [{seg_start},{seg_end}) "
                f"source_global={reason_global_source} "
                f"used_global={reason_global_used} used_local={reason_local} "
                f"dropped_missing={dropped_missing}"
            )

            reason_frames = [pil_frames[i] for i in reason_local]
            ctx_frames = [pil_frames[i] for i in context_local]

            reason_pixels = [image_transform(fr) for fr in reason_frames]
            reason_pixel_values = torch.stack(reason_pixels).unsqueeze(0).to(torch.bfloat16).cuda(non_blocking=True)
            ctx_pixels = [image_transform(fr) for fr in ctx_frames]
            ctx_pixel_values = torch.stack(ctx_pixels).unsqueeze(0).to(torch.bfloat16).cuda(non_blocking=True)

            extra_text = (
                "You are given a segment of a video, where the previous context in past segments is:\n"
                + rolling_history_summary
                + "\n\n The context is given for you to know what happened in the previous segments to maintain temporal dependencies through the video\n\n And, "
            )

            ctx_question = build_agent_question(model_context_instruct, len(ctx_frames), new_qs="", extra_text="")
            reason_question = build_agent_question(model_instruct, len(reason_frames), new_qs=new_qs, extra_text=extra_text)
            if args.print_model_output:
                log(f"reasoning_input:\n{reason_question}")

            total_attempts = max(1, int(args.label_retry_count) + 1)
            ctx_resp, reason_resp = "", ""
            label = None
            for attempt_idx in range(total_attempts):
                if attempt_idx > 0 and float(args.label_retry_wait_sec) > 0:
                    time.sleep(float(args.label_retry_wait_sec))
                if attempt_idx > 0:
                    log(
                        f"retry segment [{seg_start},{seg_end}) "
                        f"attempt={attempt_idx + 1}/{total_attempts} reason=unparsable_label"
                    )

                ctx_resp, reason_resp = batched_context_reasoning(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    ctx_pixel_values=ctx_pixel_values,
                    reason_pixel_values=reason_pixel_values,
                    ctx_question=ctx_question,
                    reason_question=reason_question,
                )
                label = extract_label_robust(reason_resp)
                if label is not None:
                    break
                preview = (reason_resp or "").replace("\n", " ")[:240]
                log(
                    f"unparsable_label segment [{seg_start},{seg_end}) "
                    f"attempt={attempt_idx + 1}/{total_attempts} preview={preview!r}"
                )

            if label is None:
                log(f"skip segment [{seg_start},{seg_end}) reason=unparsable_label")
                continue

            segments.append(
                {
                    "start": int(seg_start),
                    "end": int(seg_end),
                    "label": int(label),
                    "uniform_context_indices": context_global,
                    "mif_reasoning_indices": reason_global_used,
                }
            )

            log(f"segment [{seg_start},{seg_end}) -> label={label}")
            if args.print_model_output:
                log(f"context_output: {ctx_resp[:400]}")
                log(f"reasoning_output:\n{reason_resp}")

            if ctx_resp:
                merge_text = (
                    "Previous accumulated context summary:\n"
                    + rolling_history_summary
                    + "\n\nNew current-segment context to merge:\n"
                    + ctx_resp
                )
                rolling_history_summary = summarize_with_dummy_image(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    summarization_template=summarization_template,
                    image_transform=image_transform,
                    text=merge_text,
                )

        payload = {
            "video": video_name,
            "length": n_frms,
            "segment_len": int(args.segment_len),
            "segment_stride": int(args.segment_stride),
            "segments": segments,
            "frame_scores": src_frame_scores.tolist(),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        log(f"saved {out_json}")
        done += 1
        log(f"VIDEO_DONE video={video_name} progress={existing_done + done}/{target_total}")
        if done >= args.max_videos:
            break


if __name__ == "__main__":
    main()


