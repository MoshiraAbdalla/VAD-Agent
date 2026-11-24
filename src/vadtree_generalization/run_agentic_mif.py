from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agents import LlavaVideoClient, TransformersLLMClient
from .imagebind_selector import ImageBindFrameSelector
from .pipeline import make_config, process_dataset
from .rerun_llava_binary_reasoning import rerun_video as rerun_binary_video
from .sampling import UniformFrameSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real LLaVA + text-LLM agentic video anomaly inference.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing frames/<video_name>.")
    parser.add_argument("--dataset", choices=["ucf", "xd", "generic"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("vadtree_generalization/result/local"))
    parser.add_argument("--segment-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--context-samples", type=int, default=8)
    parser.add_argument("--reasoning-samples", type=int, default=8)
    parser.add_argument("--video-limit", type=int, default=1)
    parser.add_argument("--segment-limit", type=int, default=None)
    parser.add_argument(
        "--priority-video-list",
        type=Path,
        default=None,
        help="Optional CSV/text list of video names to process first; remaining videos continue in dataset order.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip per-video JSON files marked complete.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record video errors and continue.")
    parser.add_argument("--print-model-io", action="store_true", help="Print each model prompt/input metadata and raw output.")
    parser.add_argument("--model-io-max-chars", type=int, default=12000, help="Max chars printed per prompt/output; 0 means no clipping.")
    parser.add_argument("--summarize-first-segment", action="store_true", help="Call the narrative model on segment 0 instead of using context as initial memory.")
    parser.add_argument(
        "--narrative-prompt-variant",
        choices=["current", "coarse"],
        default="current",
        help=(
            "Narrative prompt template. Use 'coarse' to reproduce the older prompt used by "
            "vadtree_generalization/result/ucf_full_llama_narrative."
        ),
    )
    parser.add_argument(
        "--reasoning-score-mode",
        choices=["anomaly", "binary"],
        default="anomaly",
        help="Reasoning output mode: anomaly keeps the previous 0-1 JSON score; binary asks LLaVA for only 0 or 1.",
    )
    parser.add_argument(
        "--reuse-existing-agent-outputs-dir",
        type=Path,
        default=None,
        help=(
            "Directory of existing per-video VAD-Agent-MIF JSONs. With --reasoning-score-mode binary, "
            "reuse saved segment frames/memory and rerun only the LLaVA reasoning layer into --output-dir."
        ),
    )
    parser.add_argument(
        "--scenes-json",
        type=Path,
        default=None,
        help="Optional VADTree dfs_*_scenes.json. When set, VADTree scenes replace fixed-size segments.",
    )
    parser.add_argument("--llava-pretrained", type=str, required=True, help="Hugging Face ID or local path for LLaVA-Video weights.")
    parser.add_argument("--llava-model-base", type=str, default=None, help="Optional Hugging Face ID or local base model path for LoRA weights.")
    parser.add_argument("--llava-repo", type=Path, default=None, help="Optional path to a local LLaVA-NeXT repo/package.")
    parser.add_argument("--llava-model-name", type=str, default="llava_qwen")
    parser.add_argument("--conv-template", type=str, default="qwen_1_5")
    parser.add_argument("--llava-attn-implementation", type=str, default="sdpa")
    parser.add_argument("--narrative-agent", choices=["deepseek", "llava", "llama"], default="deepseek")
    parser.add_argument("--deepseek-ckpt", type=str, default=None, help="Hugging Face ID or local path for DeepSeek/HF causal-LM weights.")
    parser.add_argument("--llama-ckpt", type=str, default=r"D:\Moshira Research\models\Llama-2-13b-chat-hf", help="Hugging Face-format LLaMA checkpoint for text-only narrative memory.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--narrative-max-new-tokens", type=int, default=128)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--keep-thinking", action="store_true", help="Keep DeepSeek <think> text in narrative memory when using --narrative-agent deepseek.")
    parser.add_argument("--imagebind-repo", type=Path, default=None, help="Optional path to local ImageBind repo.")
    parser.add_argument("--imagebind-checkpoint", type=Path, default=None, help="Optional path to imagebind_huge.pth.")
    parser.add_argument("--imagebind-chunk-size", type=int, default=128)
    parser.add_argument("--imagebind-feature-dir", type=Path, default=Path("vadtree_generalization/imagebind_features"), help="Directory for reusable per-segment ImageBind feature caches.")
    parser.add_argument("--no-imagebind-feature-cache", action="store_true", help="Disable saving/loading reusable ImageBind feature files.")
    parser.add_argument("--reasoning-frame-selector", choices=["imagebind", "uniform"], default="imagebind", help="Frame selector for the LLaVA reasoning agent.")
    parser.add_argument("--mif-window", type=int, default=-1, help="Local feature-similarity window; -1 uses adaptive size.")
    parser.add_argument("--mif-alpha", type=float, default=2.5, help="Adaptive window divisor when --mif-window=-1.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], default="auto")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_8bit and args.load_4bit:
        raise ValueError("Use either --load-8bit or --load-4bit, not both.")
    print("[init] building pipeline config", flush=True)
    config = make_config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        segment_size=args.segment_size,
        stride=args.stride,
        context_samples=args.context_samples,
        reasoning_samples=args.reasoning_samples,
        video_limit=args.video_limit,
        segment_limit=args.segment_limit,
        scenes_json=args.scenes_json,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
        print_model_io=args.print_model_io,
        model_io_max_chars=args.model_io_max_chars,
        summarize_first_segment=args.summarize_first_segment,
        narrative_agent=args.narrative_agent,
        narrative_prompt_variant=args.narrative_prompt_variant,
        priority_video_list=args.priority_video_list,
        reasoning_score_mode=args.reasoning_score_mode,
    )
    print("[init] loading LLaVA context/reasoning model", flush=True)
    llava = LlavaVideoClient(
        pretrained=args.llava_pretrained,
        llava_repo=args.llava_repo,
        model_base=args.llava_model_base,
        model_name=args.llava_model_name,
        device=args.device,
        device_map=args.device_map,
        conv_template=args.conv_template,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.llava_attn_implementation,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )
    print("[init] LLaVA loaded", flush=True)
    if args.reuse_existing_agent_outputs_dir is not None:
        if args.reasoning_score_mode != "binary":
            raise ValueError("--reuse-existing-agent-outputs-dir is currently for binary reasoning reruns; use --reasoning-score-mode binary.")
        _process_reused_binary_reasoning(args, config, llava)
        print(f"Saved binary direct-score outputs to {config.output_dir.resolve()}", flush=True)
        return

    context_vlm = llava
    if args.narrative_agent == "deepseek":
        if args.deepseek_ckpt is None:
            raise ValueError("--deepseek-ckpt is required when --narrative-agent deepseek.")
        print("[init] loading DeepSeek narrative model", flush=True)
        narrative_llm = TransformersLLMClient(
            checkpoint=args.deepseek_ckpt,
            max_new_tokens=args.narrative_max_new_tokens,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            strip_thinking=not args.keep_thinking,
            force_no_think=not args.keep_thinking,
        )
        print("[init] DeepSeek loaded", flush=True)
    elif args.narrative_agent == "llava":
        print("[init] using LLaVA as narrative model; DeepSeek will not be loaded", flush=True)
        narrative_llm = None
    else:
        print("[init] loading LLaMA narrative model", flush=True)
        narrative_llm = TransformersLLMClient(
            checkpoint=args.llama_ckpt,
            max_new_tokens=args.narrative_max_new_tokens,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            strip_thinking=False,
            force_no_think=False,
            chat_format="llama2",
            system_prompt=_narrative_system_prompt(args.narrative_prompt_variant),
        )
        print("[init] LLaMA loaded", flush=True)
    reasoning_vlm = llava
    if args.reasoning_frame_selector == "uniform":
        print("[init] using uniform frame selector for reasoning; ImageBind will not be loaded", flush=True)
        frame_selector = UniformFrameSelector()
    else:
        print("[init] loading ImageBind frame selector", flush=True)
        frame_selector = ImageBindFrameSelector(
            imagebind_repo=args.imagebind_repo,
            checkpoint=args.imagebind_checkpoint,
            device=args.device,
            chunk_size=args.imagebind_chunk_size,
            window=args.mif_window,
            alpha=args.mif_alpha,
            feature_dir=None if args.no_imagebind_feature_cache else args.imagebind_feature_dir,
            dataset_name=config.dataset,
        )
        print("[init] ImageBind loaded", flush=True)
    print("[run] starting dataset processing", flush=True)
    process_dataset(config, context_vlm, narrative_llm, reasoning_vlm, frame_selector)
    print(f"Saved Agentic local-model outputs to {config.output_dir.resolve()}", flush=True)


def _narrative_system_prompt(variant: str) -> str:
    if variant == "coarse":
        return (
            "You are a concise text-only narrative memory agent for video anomaly detection. "
            "Update memory using only the provided text context. Do not use frames. "
            "Reply only with the requested four fields and no introductory text."
        )
    return (
        "You are a concise text-only narrative memory agent for video anomaly detection. "
        "Combine the previous segment story and the current segment context into one updated temporal story memory. "
        "Use only the provided text context. Do not use frames. Do not infer hidden intent. "
        "Reply only with the requested four fields and no introductory text."
    )


def _process_reused_binary_reasoning(args: argparse.Namespace, config: Any, llava: LlavaVideoClient) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _list_reusable_video_jsons(args.reuse_existing_agent_outputs_dir)
    if args.video_limit is not None:
        source_paths = source_paths[: max(0, int(args.video_limit))]

    errors: list[dict[str, str]] = []
    processed = 0
    for index, source_path in enumerate(source_paths, start=1):
        output_path = config.output_dir / source_path.name
        if args.resume and _is_complete_video_json(output_path):
            print(f"[resume] skipped completed binary video {index}: {source_path.stem}", flush=True)
            continue
        try:
            print(f"[video {index}] binary reasoning only: {source_path.stem}", flush=True)
            result = rerun_binary_video(
                source_path=source_path,
                dataset=config.dataset,
                llava=llava,
                segment_limit=args.segment_limit,
                print_model_io=args.print_model_io,
                model_io_max_chars=args.model_io_max_chars,
            )
            result["binary_directscore_config"] = {
                "source_json": str(source_path),
                "dataset": config.dataset,
                "reasoning_score_mode": "binary",
                "score_meaning": {"0": "normal", "1": "abnormal"},
            }
            if args.segment_limit is None:
                result["complete"] = True
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
        except Exception as exc:
            if not args.continue_on_error:
                raise
            errors.append({"video": source_path.stem, "error": repr(exc)})
            error_path = config.output_dir / f"{source_path.stem}.error.json"
            error_path.write_text(json.dumps(errors[-1], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[error] {source_path.stem}: {exc!r}", flush=True)

    summary = {
        "source_dir": str(args.reuse_existing_agent_outputs_dir),
        "output_dir": str(config.output_dir),
        "dataset": config.dataset,
        "processed": processed,
        "errors": errors,
    }
    (config.output_dir / "binary_directscore_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _list_reusable_video_jsons(source_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(source_dir.glob("*.json")):
        if path.name == "agentic_results.json" or path.name.startswith("metrics"):
            continue
        if "vid_score" in path.name or "refined" in path.name or "summary" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload.get("segments"), list):
            paths.append(path)
    return paths


def _is_complete_video_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("complete") is True and isinstance(payload.get("segments"), list)


if __name__ == "__main__":
    main()

