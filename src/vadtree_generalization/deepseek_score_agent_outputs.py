from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONTEXT_PROMPT = (
    "How would you rate the scene described on a scale from 0 to 1, "
    "with 0 representing a standard scene and 1 denoting a scene with "
    "suspicious or potentially criminal activities?"
)

FORMAT_PROMPT = (
    "Please provide the response in the form of a Python list and respond with only one number "
    "in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] "
    "without any textual explanation. It should begin with '[' and end with  ']'."
)

AGENT_RESPONSE_GUIDE = (
    "You are given text outputs from a multi-agent video anomaly detection pipeline.\n"
    "Agent response meanings:\n"
    "- Context Agent output: neutral visual context from uniformly sampled frames. "
    "Use it to understand scene setting, main entities, and typical activity; it is not an anomaly score.\n"
    "- Previous Narrative Memory: concise story summary of what happened before the current segment. "
    "Use it only for temporal continuity and do not invent events that are not stated.\n"
    "- Reasoning Agent explanation: visual-language explanation for the current segment, based on selected frames "
    "and anomaly-aware questions. Treat this as the main current-segment evidence.\n"
    "Score the current segment using all three text fields, with strongest weight on the current-segment "
    "reasoning explanation and temporal support from the previous narrative memory."
)


@dataclass
class ScoringConfig:
    results_dir: Path
    output_json: Path
    deepseek_ckpt: str
    dataset: str | None
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    device_map: str
    torch_dtype: str
    keep_thinking: bool
    include_incomplete: bool
    video_limit: int | None
    segment_limit: int | None
    include_llava_score: bool
    include_reasoning_response: bool
    print_model_io: bool
    model_io_max_chars: int

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["results_dir"] = str(self.results_dir)
        data["output_json"] = str(self.output_json)
        data["context_prompt"] = CONTEXT_PROMPT
        data["format_prompt"] = FORMAT_PROMPT
        data["agent_response_guide"] = AGENT_RESPONSE_GUIDE
        data["score_source"] = (
            "VAD-Agent-MIF per-coarse-scene context, narrative memory, and parsed LLaVA textual reason"
        )
        return data


class DeepSeekBatchScorer:
    def __init__(self, config: ScoringConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _resolve_torch_dtype(torch, config.torch_dtype)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.deepseek_ckpt, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.deepseek_ckpt,
            torch_dtype=dtype,
            device_map=config.device_map,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def score(self, prompts: list[str]) -> list[str]:
        dialogs = [[{"role": "user", "content": prompt}] for prompt in prompts]
        text_batch = self.tokenizer.apply_chat_template(
            dialogs,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not self.config.keep_thinking:
            text_batch = [text + "\n</think>\n\n" for text in text_batch]
        model_inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "eos_token_id": self.model.config.eos_token_id,
            "pad_token_id": self.model.config.eos_token_id,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generate_kwargs["top_k"] = self.config.top_k

        with _torch_no_grad():
            generated_ids = self.model.generate(**model_inputs, **generate_kwargs)
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score VAD-Agent-MIF coarse segment explanations with DeepSeek in VADTree vid_score format."
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="VAD-Agent-MIF run output directory with per-video JSON files.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path. Defaults under results-dir/deepseek_scores/.")
    parser.add_argument("--deepseek-ckpt", type=str, required=True, help="Hugging Face ID or local DeepSeek checkpoint.")
    parser.add_argument("--dataset", choices=["ucf", "xd", "generic"], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        default="auto",
    )
    parser.set_defaults(keep_thinking=True)
    parser.add_argument("--keep-thinking", action="store_true", help="Keep DeepSeek thinking mode, like VADTree *_think outputs. This is the default.")
    parser.add_argument("--no-thinking", action="store_false", dest="keep_thinking", help="Disable DeepSeek thinking mode for faster scoring.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing output JSON and skip already scored videos.")
    parser.add_argument("--include-incomplete", action="store_true", help="Score per-video files that are not marked complete.")
    parser.add_argument("--video-limit", type=int, default=None)
    parser.add_argument("--segment-limit", type=int, default=None)
    parser.add_argument(
        "--include-llava-score",
        action="store_true",
        help="Include the parsed LLaVA anomaly score in the DeepSeek prompt. Default is off for an independent scorer.",
    )
    parser.add_argument(
        "--include-reasoning-response",
        action="store_true",
        help="Include the full LLaVA reasoning response JSON. Default is off because it contains the LLaVA score.",
    )
    parser.add_argument("--print-model-io", action="store_true")
    parser.add_argument("--model-io-max-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.results_dir / "deepseek_scores" / "deepseek_agent_scores.json"
    config = ScoringConfig(
        results_dir=args.results_dir,
        output_json=output_json,
        deepseek_ckpt=args.deepseek_ckpt,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        keep_thinking=args.keep_thinking,
        include_incomplete=args.include_incomplete,
        video_limit=args.video_limit,
        segment_limit=args.segment_limit,
        include_llava_score=args.include_llava_score,
        include_reasoning_response=args.include_reasoning_response,
        print_model_io=args.print_model_io,
        model_io_max_chars=args.model_io_max_chars,
    )
    run(config, resume=args.resume)


def run(config: ScoringConfig, resume: bool) -> dict[str, Any]:
    video_paths = _list_video_jsons(config.results_dir)
    if config.video_limit is not None:
        video_paths = video_paths[: config.video_limit]

    output = _load_or_init_output(config, resume)
    output["config"] = config.to_json()
    output.setdefault("vid_score", {})
    config.output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] loading DeepSeek scorer: {config.deepseek_ckpt}", flush=True)
    scorer = DeepSeekBatchScorer(config)
    print("[init] DeepSeek loaded", flush=True)
    print(f"[run] scoring {len(video_paths)} video files", flush=True)

    for video_index, video_path in enumerate(video_paths, start=1):
        payload = _read_json(video_path)
        if not config.include_incomplete and payload.get("complete") is not True:
            print(f"[skip] incomplete video file: {video_path.name}", flush=True)
            continue

        video_key = _video_key(payload, video_path)
        if video_key in output["vid_score"]:
            print(f"[resume] skipped completed score {video_index}: {video_key}", flush=True)
            continue

        started = time.perf_counter()
        segments = payload.get("segments", [])
        if config.segment_limit is not None:
            segments = segments[: config.segment_limit]

        print(f"[video {video_index}] {video_key} segments={len(segments)}", flush=True)
        video_scores = _score_video_segments(scorer, config, segments)
        output["vid_score"][video_key] = video_scores
        _write_json(config.output_json, output)
        print(
            f"[video {video_index}] saved {len(video_scores)} scores in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    _write_json(config.output_json, output)
    print(f"Saved DeepSeek scores to {config.output_json.resolve()}", flush=True)
    return output


def _score_video_segments(
    scorer: DeepSeekBatchScorer,
    config: ScoringConfig,
    segments: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    video_scores: dict[str, list[Any]] = {}
    for batch_start in range(0, len(segments), config.batch_size):
        batch_segments = segments[batch_start : batch_start + config.batch_size]
        prompts = [
            _build_score_prompt(
                segment,
                include_llava_score=config.include_llava_score,
                include_reasoning_response=config.include_reasoning_response,
            )
            for segment in batch_segments
        ]
        if config.print_model_io:
            for offset, prompt in enumerate(prompts):
                print(f"\n[model-io][segment {batch_start + offset}][DeepSeek scorer][input]", flush=True)
                print(_clip_text(prompt, config.model_io_max_chars), flush=True)

        responses = scorer.score(prompts)
        for offset, (segment, response) in enumerate(zip(batch_segments, responses)):
            if config.print_model_io:
                print(f"\n[model-io][segment {batch_start + offset}][DeepSeek scorer][output]", flush=True)
                print(_clip_text(response, config.model_io_max_chars), flush=True)
            score = _parse_score(response, keep_thinking=config.keep_thinking)
            video_scores[_segment_key(segment)] = [score, response]

    return video_scores


def _build_score_prompt(
    segment: dict[str, Any],
    include_llava_score: bool = False,
    include_reasoning_response: bool = False,
) -> str:
    context = _clean(segment.get("context"))
    narrative_memory = _clean(segment.get("narrative_memory"))
    reasoning_response = _clean(segment.get("reasoning_response"))
    explanation = _clean(segment.get("explanation"))
    llava_score = segment.get("anomaly_score")

    scene_description = (
        f"{AGENT_RESPONSE_GUIDE}\n\n"
        "Context Agent output:\n"
        f"{context or 'none'}\n\n"
        "Previous Narrative Memory:\n"
        f"{narrative_memory or 'none'}\n\n"
        "Reasoning Agent explanation:\n"
        f"{explanation or 'none'}"
    )
    if include_reasoning_response:
        scene_description += (
            "\n\nFull reasoning agent response:\n"
            f"{reasoning_response or 'none'}"
        )
    if include_llava_score:
        scene_description += (
            "\n\nInitial LLaVA anomaly score:\n"
            f"{llava_score if llava_score is not None else 'none'}"
        )
    return f"{CONTEXT_PROMPT} {FORMAT_PROMPT}\n\n{scene_description}"


def _parse_score(response: str, keep_thinking: bool) -> float:
    text = response.split("</think>")[-1] if keep_thinking else response
    match = re.search(r"\[(\d+(?:\.\d+)?)\]", text)
    if not match:
        print(f"\t<Warning! Response did not follow score format:> {_clip_text(response, 500)}", flush=True)
        return -1.0
    score = float(match.group(1))
    allowed = {i / 10 for i in range(11)}
    if score not in allowed:
        print(f"\t<Warning! Score {score} is outside the requested discrete list.>", flush=True)
    return score


def _list_video_jsons(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    paths = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "agentic_results.json":
            continue
        try:
            payload = _read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("segments"), list):
            paths.append(path)
    return paths


def _load_or_init_output(config: ScoringConfig, resume: bool) -> dict[str, Any]:
    if resume and config.output_json.exists():
        return _read_json(config.output_json)
    if config.output_json.exists() and not resume:
        raise FileExistsError(f"Output exists. Use --resume or choose another --output-json: {config.output_json}")
    return {"config": config.to_json(), "vid_score": {}}


def _video_key(payload: dict[str, Any], video_path: Path) -> str:
    name = str(payload.get("video_name") or video_path.stem)
    return name if name.endswith(".mp4") else f"{name}.mp4"


def _segment_key(segment: dict[str, Any]) -> str:
    plan = segment.get("plan", {})
    return f"{int(plan['start'])}, {int(plan['end'])}"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[... clipped {omitted} chars ...]"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_torch_dtype(torch_module, dtype: str):
    if dtype == "auto":
        return torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32
    if dtype in {"float16", "fp16"}:
        return torch_module.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


class _torch_no_grad:
    def __enter__(self):
        import torch

        self.context = torch.no_grad()
        return self.context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.context.__exit__(exc_type, exc_val, exc_tb)


if __name__ == "__main__":
    main()

