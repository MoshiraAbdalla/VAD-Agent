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
from transformers import AutoModel, AutoTokenizer, GitVisionModel

from src.vad_detection.utils.frame_selection import iter_snippet_frames, sample_representative_frames_MIF


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
    return T.Compose([
        EnsureRGB(),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


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


def uniform_indices(length: int, k: int):
    if length <= 0:
        return []
    k = max(1, min(k, length))
    return np.linspace(0, length - 1, num=k, dtype=int).tolist()


def smooth_scores_gaussian(arr: np.ndarray, win: int = 31, sigma: float = 5.0) -> np.ndarray:
    if arr.size == 0 or win <= 1:
        return arr.astype(np.float32, copy=False)
    pad = win // 2
    x = np.arange(-pad, pad + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= np.sum(kernel) + 1e-8
    padded = np.pad(arr, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")[pad:-pad]
    return smoothed.astype(np.float32)


def compute_motion_proxy_scores_from_feats_v3(feats):
    if feats is None:
        return np.asarray([], dtype=np.float32)
    if isinstance(feats, torch.Tensor):
        f = feats.float().cpu()
    else:
        f = torch.as_tensor(feats, dtype=torch.float32)
    if f.numel() == 0:
        return np.asarray([], dtype=np.float32)

    f = torch.nn.functional.normalize(f, dim=1)
    if f.size(0) == 1:
        return np.asarray([], dtype=np.float32)

    sim = torch.sum(f[1:] * f[:-1], dim=1).clamp(-1, 1)
    motion = 1.0 - sim
    motion = torch.cat([torch.zeros(1), motion], dim=0).numpy().astype(np.float32)

    mn = float(np.min(motion))
    mx = float(np.max(motion))
    if mx - mn < 1e-8:
        return motion.astype(np.float32)
    else:
        norm = (motion - mn) / (mx - mn + 1e-8)
    return smooth_scores_gaussian(norm, win=31, sigma=5.0)


def segment_feature_from_frame_feats(frame_feats):
    if frame_feats is None:
        return None
    if isinstance(frame_feats, torch.Tensor):
        ff = frame_feats.float().cpu()
    else:
        ff = torch.as_tensor(frame_feats, dtype=torch.float32)
    if ff.numel() == 0:
        return None
    seg = ff.mean(dim=0, keepdim=True)
    seg = torch.nn.functional.normalize(seg, dim=1)
    return seg[0].numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-root", default="data/ucf/frames/")
    parser.add_argument("--ann-root", default="data/annotations/UCF_Eval.json")
    parser.add_argument("--save-folder", default="results/predictions/ucf_crime_mif_stride150/")
    parser.add_argument("--segment-feature-cache-dir", default="results/cached_features/ucf_crime_stride150/")
    parser.add_argument(
        "--model-path",
        default="OpenGVLab/InternVL2-8B",
        help="Hugging Face model id or local path for InternVL2-8B.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Hugging Face cache directory for InternVL2 and GIT models.",
    )
    parser.add_argument(
        "--disable-segment-feature-save",
        action="store_true",
        help="Do not save per-video segment feature cache (.npy); resume checks use JSON only.",
    )
    parser.add_argument("--segment-len", type=int, default=300)
    parser.add_argument("--segment-stride", type=int, default=150)
    parser.add_argument("--reasoning-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=16)
    parser.add_argument("--mif-w", type=int, default=8)
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
    os.makedirs(args.segment_feature_cache_dir, exist_ok=True)
    log("deterministic mode enabled; starting UCF MIF-segment-motion-proxy-v4 (300/150 overlap) run")

    with open("configs/prompts/reasoning_agent.txt", "r", encoding="utf-8") as f:
        model_instruct = f.read()
    with open("configs/prompts/context_agent.txt", "r", encoding="utf-8") as f:
        model_context_instruct = f.read()
    with open("configs/prompts/narrative_agent.txt", "r", encoding="utf-8") as f:
        summarization_template = f.read()

    # UCF-Crime guiding questions
    new_qs = (
        "1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior?\n"
        "2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?\n"
        "3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?\n"
        "4. Is there any visible damage or unusual movement in the video that indicates an anomaly?\n"
        "5. Are there any unusual sounds or noises in the video that suggest an anomaly?\n"
    )

    path = args.model_path
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        cache_dir=args.cache_dir,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=args.cache_dir)

    git_model_name = "microsoft/git-base"
    git_model = GitVisionModel.from_pretrained(git_model_name, cache_dir=args.cache_dir).eval().to("cuda")
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
        out_feat_i = os.path.join(args.segment_feature_cache_dir, f"{vn}.npy")
        done_has_json = os.path.exists(out_json_i)
        done_has_feat = os.path.exists(out_feat_i)
        if done_has_json and (args.disable_segment_feature_save or done_has_feat):
            existing_done += 1
    log(f"resume status: already_done={existing_done}/{target_total}")

    done = 0
    for video_path, video_name, n_frms in loader:
        video_path = video_path[0]
        video_name = video_name[0]
        n_frms = int(n_frms.item())

        out_json = os.path.join(args.save_folder, f"{video_name}.json")
        out_feat = os.path.join(args.segment_feature_cache_dir, f"{video_name}.npy")
        done_has_json = os.path.exists(out_json)
        done_has_feat = os.path.exists(out_feat)
        if done_has_json and (args.disable_segment_feature_save or done_has_feat):
            continue

        log(f"processing video={video_name} frames={n_frms}")
        rolling_history_summary = "No prior context, this is the first segment of the video."
        frame_sum = np.zeros((n_frms,), dtype=np.float32)
        frame_cnt = np.zeros((n_frms,), dtype=np.float32)
        segments = []
        seg_features = []

        for seg_start in range(0, n_frms, args.segment_stride):
            seg_end = min(seg_start + args.segment_len, n_frms)
            pil_frames, frame_ids = iter_snippet_frames(
                video_path,
                seg_start,
                seg_end,
                fps_downsample=1,
                return_indices=True,
            )
            if len(pil_frames) < 2:
                log(f"skip segment [{seg_start},{seg_end}) reason=too_few_frames")
                continue

            reasoning_k = min(args.reasoning_k, len(pil_frames))
            mif_indices_local, mif_feats = sample_representative_frames_MIF(
                pil_frames,
                model=git_model,
                K=reasoning_k,
                W=args.mif_w,
                CHUNK_SIZE=128,
                return_feats=True,
            )
            mif_indices_local = sorted(mif_indices_local)
            reason_local = mif_indices_local
            context_local = uniform_indices(len(pil_frames), args.context_k)

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
            # Keep feature rows aligned with saved segment labels.
            seg_feat = segment_feature_from_frame_feats(mif_feats)
            segments.append(
                {
                    "start": int(seg_start),
                    "end": int(seg_end),
                    "label": int(label),
                    "uniform_context_indices": [int(frame_ids[i]) for i in context_local],
                    "mif_reasoning_indices": [int(frame_ids[i]) for i in reason_local],
                }
            )
            seg_features.append(seg_feat)

            log(f"segment [{seg_start},{seg_end}) -> label={label}")
            if args.print_model_output:
                log(f"context_output: {ctx_resp[:400]}")
                log(f"reasoning_output:\n{reason_resp}")

            summary_resp = ""
            if ctx_resp:
                merge_text = (
                    "Previous accumulated context summary:\n"
                    + rolling_history_summary
                    + "\n\nNew current-segment context to merge:\n"
                    + ctx_resp
                )
                summary_resp = summarize_with_dummy_image(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    summarization_template=summarization_template,
                    image_transform=image_transform,
                    text=merge_text,
                )
                rolling_history_summary = summary_resp

            segments[-1]["agent_outputs"] = {
                "context_agent": ctx_resp,
                "reasoning_agent": reason_resp,
                "summarization_agent": summary_resp,
            }

            motion_norm = compute_motion_proxy_scores_from_feats_v3(mif_feats)
            if motion_norm.size != len(frame_ids):
                log(
                    f"skip motion fill for segment [{seg_start},{seg_end}) "
                    f"reason=motion_len_mismatch motion_len={motion_norm.size} frames={len(frame_ids)}"
                )
                continue
            for li, gi in enumerate(frame_ids):
                gi = int(gi)
                if 0 <= gi < n_frms:
                    frame_sum[gi] += float(motion_norm[li])
                    frame_cnt[gi] += 1.0

        frame_scores = np.zeros((n_frms,), dtype=np.float32)
        valid = frame_cnt > 0
        frame_scores[valid] = (frame_sum[valid] / np.maximum(frame_cnt[valid], 1e-8)).astype(np.float32)

        payload = {
            "video": video_name,
            "length": n_frms,
            "segment_len": int(args.segment_len),
            "segment_stride": int(args.segment_stride),
            "segments": segments,
            "frame_scores": frame_scores.tolist(),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        # strict: save only valid segment features, no nearest-fill fallback
        if args.disable_segment_feature_save:
            seg_arr = None
            log(f"feature save disabled for {video_name}")
        else:
            valid_feats = [sf for sf in seg_features if sf is not None]
            if not valid_feats:
                log(f"skip feature save for {video_name}: no valid segment features")
                seg_arr = np.asarray([], dtype=np.float32).reshape(0, 0)
            else:
                seg_arr = np.stack(valid_feats).astype(np.float32)
            np.save(out_feat, seg_arr)

        log(f"saved {out_json}")
        if seg_arr is not None:
            log(f"saved {out_feat} shape={seg_arr.shape}")
        done += 1
        log(f"VIDEO_DONE video={video_name} progress={existing_done + done}/{target_total}")
        if done >= args.max_videos:
            break


if __name__ == "__main__":
    main()




