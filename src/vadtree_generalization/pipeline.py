from __future__ import annotations

import json
import time
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from .agents import LLMClient, VLMClient, parse_binary_reasoning, parse_reasoning_json
from .frame_store import infer_dataset_name, iter_video_dirs, list_frame_paths
from .prompts import (
    build_context_prompt,
    build_initial_narrative_memory,
    build_narrative_prompt_variant,
    build_reasoning_prompt,
    normalize_narrative_memory,
)
from .sampling import FrameSelector, build_segment_plans
from .scene_store import SceneMap, get_video_scenes, load_vadtree_scenes, vadtree_scenes_to_inclusive_ranges
from .schemas import SegmentPlan, SegmentResult, VideoResult


DEFAULT_REASONING_PARSE_RETRIES = 2


@dataclass
class PipelineConfig:
    dataset: str
    dataset_root: Path
    output_dir: Path
    segment_size: int = 64
    stride: int | None = None
    context_samples: int = 8
    reasoning_samples: int = 8
    video_limit: int | None = None
    segment_limit: int | None = None
    scenes_json: Path | None = None
    resume: bool = False
    continue_on_error: bool = False
    print_model_io: bool = False
    model_io_max_chars: int = 12000
    summarize_first_segment: bool = False
    narrative_agent: str = "deepseek"
    narrative_prompt_variant: str = "current"
    priority_video_list: Path | None = None
    reasoning_score_mode: str = "anomaly"

    def to_json(self) -> Dict[str, object]:
        data = asdict(self)
        data["dataset_root"] = str(self.dataset_root)
        data["output_dir"] = str(self.output_dir)
        data["scenes_json"] = str(self.scenes_json) if self.scenes_json is not None else None
        data["priority_video_list"] = str(self.priority_video_list) if self.priority_video_list is not None else None
        return data


def process_video(
    video_dir: Path,
    config: PipelineConfig,
    context_vlm: VLMClient,
    narrative_llm: LLMClient | None,
    reasoning_vlm: VLMClient,
    frame_selector: FrameSelector,
    scene_map: SceneMap | None = None,
    output_path: Path | None = None,
) -> VideoResult:
    frame_paths = list_frame_paths(video_dir)
    ranges = None
    if scene_map is not None:
        scenes = get_video_scenes(scene_map, video_dir.name)
        if scenes is None:
            raise ValueError(f"No VADTree scenes found for {video_dir.name}")
        ranges = vadtree_scenes_to_inclusive_ranges(scenes, len(frame_paths))

    plans = build_segment_plans(
        frame_paths=frame_paths,
        segment_size=config.segment_size,
        context_samples=config.context_samples,
        reasoning_samples=config.reasoning_samples,
        frame_selector=frame_selector,
        stride=config.stride,
        ranges=ranges,
    )
    if config.segment_limit is not None:
        plans = plans[: config.segment_limit]

    memory = ""
    segment_results: List[SegmentResult] = []
    start_segment_idx = 0
    if config.resume and output_path is not None and output_path.exists():
        segment_results, memory, start_segment_idx = _load_resumable_segments(
            output_path,
            video_name=video_dir.name,
            num_frames=len(frame_paths),
            plans=plans,
        )
        if start_segment_idx:
            print(
                f"[resume] loaded {start_segment_idx}/{len(plans)} saved segments for {video_dir.name}; "
                f"continuing at segment {start_segment_idx}",
                flush=True,
            )

    for plan in plans[start_segment_idx:]:
        segment_started = time.perf_counter()
        context_prompt = build_context_prompt(config.dataset, plan)
        if config.print_model_io:
            _print_model_input("LLaVA context agent", plan.segment_id, context_prompt, plan.context_frames, config.model_io_max_chars)
        started = time.perf_counter()
        context = context_vlm.generate(
            context_prompt,
            [frame.path for frame in plan.context_frames],
        )
        context_seconds = time.perf_counter() - started
        if config.print_model_io:
            _print_model_output("LLaVA context agent", plan.segment_id, context, config.model_io_max_chars)

        previous_memory = memory
        if not memory and not config.summarize_first_segment:
            narrative_prompt = "[skipped] First segment has no previous memory; current context is used as the initial story memory."
            memory = build_initial_narrative_memory(context)
            narrative_seconds = 0.0
            if config.print_model_io:
                _print_model_output(_narrative_agent_name(config), plan.segment_id, memory, config.model_io_max_chars)
        else:
            narrative_prompt = build_narrative_prompt_variant(memory, context, config.narrative_prompt_variant)
            if config.print_model_io:
                _print_model_input(_narrative_agent_name(config), plan.segment_id, narrative_prompt, _narrative_frames(config, plan), config.model_io_max_chars)
            started = time.perf_counter()
            if config.narrative_agent == "llava":
                memory = context_vlm.generate(
                    narrative_prompt,
                    [frame.path for frame in plan.context_frames],
                ).strip()
            else:
                if narrative_llm is None:
                    raise ValueError("A text narrative agent was selected but no LLM client was provided.")
                memory = narrative_llm.generate(narrative_prompt).strip()
            if config.print_model_io:
                _print_model_output(f"{_narrative_agent_name(config)} raw", plan.segment_id, memory, config.model_io_max_chars)
            memory = normalize_narrative_memory(memory, previous_memory, context)
            narrative_seconds = time.perf_counter() - started
            if config.print_model_io:
                _print_model_output(f"{_narrative_agent_name(config)} normalized", plan.segment_id, memory, config.model_io_max_chars)

        reasoning_prompt = build_reasoning_prompt(config.dataset, previous_memory, plan, config.reasoning_score_mode)
        if config.print_model_io:
            _print_model_input("LLaVA reasoning agent", plan.segment_id, reasoning_prompt, plan.selected_frames, config.model_io_max_chars)
        reasoning_text, score, explanation, reasoning_seconds, reasoning_trials = _generate_reasoning_with_retries(
            reasoning_vlm=reasoning_vlm,
            prompt=reasoning_prompt,
            frame_paths=[frame.path for frame in plan.selected_frames],
            score_mode=config.reasoning_score_mode,
            segment_id=plan.segment_id,
            print_model_io=config.print_model_io,
            model_io_max_chars=config.model_io_max_chars,
        )
        segment_seconds = time.perf_counter() - segment_started
        print(
            f"[timing][segment {plan.segment_id}] "
            f"context={context_seconds:.1f}s narrative={narrative_seconds:.1f}s "
            f"reasoning={reasoning_seconds:.1f}s total={segment_seconds:.1f}s",
            flush=True,
        )

        segment_results.append(
            SegmentResult(
                plan=plan,
                context_prompt=context_prompt,
                context=context,
                narrative_prompt=narrative_prompt,
                narrative_memory=memory,
                reasoning_prompt=reasoning_prompt,
                reasoning_response=reasoning_text,
                anomaly_score=score,
                explanation=explanation,
                timing_seconds={
                    "context": round(context_seconds, 4),
                    "narrative": round(narrative_seconds, 4),
                    "reasoning": round(reasoning_seconds, 4),
                    "total": round(segment_seconds, 4),
                    "reasoning_trials": reasoning_trials,
                },
            )
        )
        if output_path is not None:
            partial = VideoResult(video_name=video_dir.name, num_frames=len(frame_paths), segments=segment_results)
            _write_video_json(output_path, partial, complete=False)

    return VideoResult(video_name=video_dir.name, num_frames=len(frame_paths), segments=segment_results)


def process_dataset(
    config: PipelineConfig,
    context_vlm: VLMClient,
    narrative_llm: LLMClient | None,
    reasoning_vlm: VLMClient,
    frame_selector: FrameSelector,
) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scene_map = load_vadtree_scenes(config.scenes_json) if config.scenes_json is not None else None
    results: Dict[str, object] = {"config": config.to_json(), "videos": {}, "errors": {}}
    processed = 0
    video_dirs = list(iter_video_dirs(config.dataset_root, limit=None if scene_map is not None else config.video_limit))
    if config.priority_video_list is not None:
        video_dirs = _prioritize_video_dirs(video_dirs, config.priority_video_list)
    for video_dir in video_dirs:
        if scene_map is not None and get_video_scenes(scene_map, video_dir.name) is None:
            continue
        if config.video_limit is not None and processed >= config.video_limit:
            break

        per_video_path = config.output_dir / f"{video_dir.name}.json"
        if config.resume and per_video_path.exists():
            saved = _read_json(per_video_path)
            if saved.get("complete") is True:
                results["videos"][video_dir.name] = saved
                processed += 1
                print(f"[resume] skipped completed video {processed}: {video_dir.name}", flush=True)
                continue

        print(f"[video {processed + 1}] {video_dir.name}", flush=True)
        try:
            video_result = process_video(
                video_dir,
                config,
                context_vlm,
                narrative_llm,
                reasoning_vlm,
                frame_selector,
                scene_map=scene_map,
                output_path=per_video_path,
            )
        except Exception as exc:
            if not config.continue_on_error:
                raise
            results["errors"][video_dir.name] = repr(exc)
            processed += 1
            all_path = config.output_dir / "agentic_results.json"
            all_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue

        final_payload = _write_video_json(per_video_path, video_result, complete=True)
        results["videos"][video_dir.name] = final_payload
        processed += 1

    all_path = config.output_dir / "agentic_results.json"
    all_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def make_config(
    dataset_root: Path,
    output_dir: Path,
    dataset: str | None = None,
    segment_size: int = 64,
    stride: int | None = None,
    context_samples: int = 8,
    reasoning_samples: int = 8,
    video_limit: int | None = None,
    segment_limit: int | None = None,
    scenes_json: Path | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
    print_model_io: bool = False,
    model_io_max_chars: int = 12000,
    summarize_first_segment: bool = False,
    narrative_agent: str = "deepseek",
    narrative_prompt_variant: str = "current",
    priority_video_list: Path | None = None,
    reasoning_score_mode: str = "anomaly",
) -> PipelineConfig:
    resolved_dataset = dataset or infer_dataset_name(str(dataset_root))
    return PipelineConfig(
        dataset=resolved_dataset,
        dataset_root=dataset_root,
        output_dir=output_dir,
        segment_size=segment_size,
        stride=stride,
        context_samples=context_samples,
        reasoning_samples=reasoning_samples,
        video_limit=video_limit,
        segment_limit=segment_limit,
        scenes_json=scenes_json,
        resume=resume,
        continue_on_error=continue_on_error,
        print_model_io=print_model_io,
        model_io_max_chars=model_io_max_chars,
        summarize_first_segment=summarize_first_segment,
        narrative_agent=narrative_agent,
        narrative_prompt_variant=narrative_prompt_variant,
        priority_video_list=priority_video_list,
        reasoning_score_mode=reasoning_score_mode,
    )


def _write_video_json(path: Path, video_result: VideoResult, complete: bool) -> Dict[str, Any]:
    payload = video_result.to_json()
    payload["complete"] = complete
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prioritize_video_dirs(video_dirs: list[Path], priority_video_list: Path) -> list[Path]:
    priority_names = _load_priority_video_names(priority_video_list)
    if not priority_names:
        return video_dirs
    index = {path.name: path for path in video_dirs}
    ordered: list[Path] = []
    seen: set[str] = set()
    for name in priority_names:
        path = index.get(name)
        if path is None or name in seen:
            continue
        ordered.append(path)
        seen.add(name)
    ordered.extend(path for path in video_dirs if path.name not in seen)
    print(
        f"[priority] loaded {len(priority_names)} requested videos from {priority_video_list}; "
        f"matched {len(seen)} frame directories",
        flush=True,
    )
    return ordered


def _load_priority_video_names(path: Path) -> list[str]:
    names: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            if "," in sample:
                reader = csv.DictReader(f)
                for row in reader:
                    value = row.get("name") or row.get("video") or next(iter(row.values()), "")
                    normalized = _normalize_video_list_name(value)
                    if normalized:
                        names.append(normalized)
            else:
                for line in f:
                    normalized = _normalize_video_list_name(line.strip())
                    if normalized:
                        names.append(normalized)
    except OSError as exc:
        raise ValueError(f"Could not read --priority-video-list {path}: {exc}") from exc
    return names


def _normalize_video_list_name(value: object) -> str:
    text = str(value or "").strip().strip('"')
    if not text:
        return ""
    name = Path(text).name
    if name.lower().endswith(".mp4"):
        return name[:-4]
    return name


def _load_resumable_segments(
    path: Path,
    video_name: str,
    num_frames: int,
    plans: list[SegmentPlan],
) -> tuple[list[SegmentResult], str, int]:
    try:
        saved = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return [], "", 0
    if saved.get("complete") is True:
        return [], "", 0
    if saved.get("video_name") != video_name or int(saved.get("num_frames", -1)) != int(num_frames):
        return [], "", 0

    restored: list[SegmentResult] = []
    memory = ""
    for segment_payload in saved.get("segments", []):
        if len(restored) >= len(plans):
            break
        plan = plans[len(restored)]
        payload_plan = segment_payload.get("plan", {})
        if not _segment_plan_matches(payload_plan, plan):
            break
        restored_segment = _segment_result_from_json(segment_payload, plan)
        restored.append(restored_segment)
        memory = restored_segment.narrative_memory or memory
    return restored, memory, len(restored)


def _segment_plan_matches(payload_plan: dict[str, Any], plan: SegmentPlan) -> bool:
    try:
        return (
            int(payload_plan.get("segment_id")) == int(plan.segment_id)
            and int(payload_plan.get("start")) == int(plan.start)
            and int(payload_plan.get("end")) == int(plan.end)
        )
    except (TypeError, ValueError):
        return False


def _segment_result_from_json(payload: dict[str, Any], plan: SegmentPlan) -> SegmentResult:
    score = payload.get("anomaly_score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    timing = payload.get("timing_seconds") if isinstance(payload.get("timing_seconds"), dict) else {}
    return SegmentResult(
        plan=plan,
        context_prompt=str(payload.get("context_prompt") or ""),
        context=str(payload.get("context") or ""),
        narrative_prompt=str(payload.get("narrative_prompt") or ""),
        narrative_memory=str(payload.get("narrative_memory") or ""),
        reasoning_prompt=str(payload.get("reasoning_prompt") or ""),
        reasoning_response=str(payload.get("reasoning_response") or ""),
        anomaly_score=score,
        explanation=str(payload.get("explanation") or ""),
        timing_seconds={str(k): float(v) for k, v in timing.items() if isinstance(v, (int, float))},
    )


def _print_model_input(agent_name: str, segment_id: int, prompt: str, frames: list, max_chars: int) -> None:
    frame_indices = [getattr(frame, "index", None) for frame in frames]
    print(f"\n[model-io][segment {segment_id}][{agent_name}][input]", flush=True)
    if frame_indices:
        print(f"frame_indices: {frame_indices}", flush=True)
    else:
        print("frame_indices: none", flush=True)
    print("prompt:", flush=True)
    print(_clip_text(prompt, max_chars), flush=True)


def _print_model_output(agent_name: str, segment_id: int, output: str, max_chars: int) -> None:
    print(f"\n[model-io][segment {segment_id}][{agent_name}][output]", flush=True)
    print(_clip_text(output, max_chars), flush=True)


def _generate_reasoning_with_retries(
    reasoning_vlm: VLMClient,
    prompt: str,
    frame_paths: list[Path],
    score_mode: str,
    segment_id: int,
    print_model_io: bool,
    model_io_max_chars: int,
) -> tuple[str, float | None, str, float, int]:
    last_text = ""
    last_score: float | None = None
    last_explanation = ""
    total_seconds = 0.0
    for trial in range(1, DEFAULT_REASONING_PARSE_RETRIES + 1):
        started = time.perf_counter()
        text = reasoning_vlm.generate(prompt, frame_paths)
        total_seconds += time.perf_counter() - started
        if score_mode == "binary":
            score, explanation = parse_binary_reasoning(text)
        else:
            score, explanation = parse_reasoning_json(text)
        last_text = text
        last_score = score
        last_explanation = explanation
        if print_model_io:
            suffix = "" if trial == 1 else f" retry {trial}"
            _print_model_output(
                f"LLaVA reasoning agent{suffix}",
                segment_id,
                f"{text}\n[parsed_score] {score}",
                model_io_max_chars,
            )
        if score is not None:
            return text, score, explanation, total_seconds, trial
    return last_text, last_score, last_explanation or last_text, total_seconds, DEFAULT_REASONING_PARSE_RETRIES


def _narrative_agent_name(config: PipelineConfig) -> str:
    if config.narrative_agent == "llava":
        return "LLaVA narrative agent"
    if config.narrative_agent == "llama":
        return "LLaMA narrative agent"
    return "DeepSeek narrative agent"


def _narrative_frames(config: PipelineConfig, plan) -> list:
    if config.narrative_agent == "llava":
        return plan.context_frames
    return []


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[... clipped {omitted} chars; set --model-io-max-chars 0 for full text ...]"

