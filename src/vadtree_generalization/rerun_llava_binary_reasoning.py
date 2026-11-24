from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .agents import LlavaVideoClient, parse_binary_reasoning
from .prompts import build_binary_reasoning_prompt


DEFAULT_SOURCE_DIR = Path("vadtree_generalization/result/ucf_fine_mif8_ctx16_llama_narrative_adjusted")
DEFAULT_OUTPUT_DIR = Path("vadtree_generalization/result/ucf_fine_mif8_ctx16_directscore")
DEFAULT_BINARY_PARSE_RETRIES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse saved VAD-Agent-MIF context/narrative/frame selections and rerun only "
            "the LLaVA reasoning layer with a strict binary 0/1 output."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", choices=["ucf", "xd", "generic"], required=True)
    parser.add_argument("--video-limit", type=int, default=None)
    parser.add_argument("--segment-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--print-model-io", action="store_true")
    parser.add_argument("--model-io-max-chars", type=int, default=8000)
    parser.add_argument("--llava-pretrained", type=str, required=True)
    parser.add_argument("--llava-model-base", type=str, default=None)
    parser.add_argument("--llava-repo", type=Path, default=None)
    parser.add_argument("--llava-model-name", type=str, default="llava_qwen")
    parser.add_argument("--conv-template", type=str, default="qwen_1_5")
    parser.add_argument("--llava-attn-implementation", type=str, default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=8)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = _list_source_videos(args.source_dir)
    if args.video_limit is not None:
        videos = videos[: max(0, int(args.video_limit))]

    print("[init] loading LLaVA binary reasoning model", flush=True)
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

    errors: list[dict[str, Any]] = []
    processed = 0
    for index, source_path in enumerate(videos, start=1):
        output_path = args.output_dir / source_path.name
        if args.resume and _is_complete(output_path):
            print(f"[resume] skipped completed video {index}: {source_path.stem}", flush=True)
            continue
        try:
            print(f"[video {index}] {source_path.stem}", flush=True)
            result = rerun_video(
                source_path=source_path,
                dataset=args.dataset,
                llava=llava,
                segment_limit=args.segment_limit,
                print_model_io=args.print_model_io,
                model_io_max_chars=args.model_io_max_chars,
            )
            result["binary_directscore_config"] = {
                "source_json": str(source_path),
                "dataset": args.dataset,
                "prompt_type": "strict_binary_0_1",
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
            partial_path = args.output_dir / f"{source_path.stem}.error.json"
            partial_path.write_text(json.dumps(errors[-1], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[error] {source_path.stem}: {exc!r}", flush=True)

    summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "dataset": args.dataset,
        "processed": processed,
        "errors": errors,
    }
    (args.output_dir / "binary_directscore_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved binary direct-score outputs to {args.output_dir.resolve()}", flush=True)


def rerun_video(
    source_path: Path,
    dataset: str,
    llava: LlavaVideoClient,
    segment_limit: int | None,
    print_model_io: bool,
    model_io_max_chars: int,
) -> dict[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    segments = list(payload.get("segments") or [])
    if segment_limit is not None:
        segments = segments[: max(0, int(segment_limit))]

    out = dict(payload)
    out_segments = []
    previous_memory = ""
    for idx, segment in enumerate(segments):
        plan = segment.get("plan") or {}
        selected_frames = _selected_frame_paths(plan)
        if not selected_frames:
            raise ValueError(f"segment {idx} has no selected frame paths")

        prompt = build_binary_reasoning_prompt(dataset=dataset, previous_memory=previous_memory)
        if print_model_io:
            print(_clip(f"[binary input segment {idx}]\n{prompt}\nframes={selected_frames}", model_io_max_chars), flush=True)

        response, score, explanation, elapsed, trial = _generate_binary_with_retries(
            llava=llava,
            prompt=prompt,
            selected_frames=selected_frames,
            segment_index=idx,
            print_model_io=print_model_io,
            model_io_max_chars=model_io_max_chars,
        )
        parse_error = None
        if score is None:
            parse_error = f"Could not parse binary LLaVA response: {response!r}"
            explanation = response

        updated = dict(segment)
        for key in (
            "original_reasoning_prompt",
            "original_reasoning_response",
            "original_anomaly_score",
            "original_explanation",
        ):
            updated.pop(key, None)
        updated["reasoning_prompt"] = prompt
        updated["reasoning_response"] = response
        updated["anomaly_score"] = int(score) if score is not None else None
        updated["explanation"] = explanation
        if parse_error is not None:
            updated["parse_error"] = parse_error
        timing = dict(updated.get("timing_seconds") or {})
        timing["binary_reasoning"] = round(elapsed, 4)
        timing["binary_reasoning_trials"] = trial
        updated["timing_seconds"] = timing
        model_outputs = dict(updated.get("model_outputs") or {})
        model_outputs["binary_reasoning_agent"] = response
        model_outputs["reasoning_agent"] = response
        updated["model_outputs"] = model_outputs
        out_segments.append(updated)

        previous_memory = str(segment.get("narrative_memory") or previous_memory)

    out["segments"] = out_segments
    if segment_limit is not None and len(out_segments) < len(payload.get("segments") or []):
        out["complete"] = False
        out["segment_limit"] = segment_limit
    return out


def _generate_binary_with_retries(
    llava: LlavaVideoClient,
    prompt: str,
    selected_frames: list[Path],
    segment_index: int,
    print_model_io: bool,
    model_io_max_chars: int,
) -> tuple[str, float | None, str, float, int]:
    last_response = ""
    last_score: float | None = None
    last_explanation = ""
    elapsed_total = 0.0
    for trial in range(1, DEFAULT_BINARY_PARSE_RETRIES + 1):
        started = time.perf_counter()
        response = llava.generate(prompt, selected_frames).strip()
        elapsed_total += time.perf_counter() - started
        score, explanation = parse_binary_reasoning(response)
        last_response = response
        last_score = score
        last_explanation = explanation
        if print_model_io:
            print(
                _clip(f"[binary output segment {segment_index} trial {trial}] {response} -> {score}", model_io_max_chars),
                flush=True,
            )
        if score is not None:
            return response, score, explanation, elapsed_total, trial
    return last_response, last_score, last_explanation or last_response, elapsed_total, DEFAULT_BINARY_PARSE_RETRIES


def _selected_frame_paths(plan: dict[str, Any]) -> list[Path]:
    paths = []
    for frame in plan.get("selected_frames") or []:
        path = frame.get("path") if isinstance(frame, dict) else None
        if path:
            paths.append(Path(path))
    return paths


def _list_source_videos(source_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(source_dir.glob("*.json")):
        if path.name == "agentic_results.json" or path.name.startswith("metrics"):
            continue
        if "vid_score" in path.name or "refined" in path.name or "summary" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("segments"), list):
            paths.append(path)
    return paths


def _is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("complete") is True and isinstance(payload.get("segments"), list)


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[clipped]"


if __name__ == "__main__":
    main()

