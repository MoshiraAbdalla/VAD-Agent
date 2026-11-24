from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agents import parse_reasoning_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VAD-Agent-MIF per-video JSONs to VADTree vid_score format.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing per-video Agentic JSONs.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--complete-only", action="store_true", help="Only export videos marked complete.")
    parser.add_argument("--video-suffix", type=str, default=".mp4", help="Suffix appended to stem-only video names.")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Store [score, explanation, reasoning_response] instead of just [score, explanation].",
    )
    parser.add_argument(
        "--missing-score",
        type=float,
        default=0.0,
        help="Fallback score for segments whose model response cannot be parsed; keeps segment alignment intact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_vid_score(
        results_dir=args.results_dir,
        complete_only=args.complete_only,
        video_suffix=args.video_suffix,
        include_text=args.include_text,
        missing_score=args.missing_score,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output_json.resolve()}")
    print(
        f"videos={len(payload['vid_score'])} "
        f"segments={sum(len(v) for v in payload['vid_score'].values())}"
    )


def export_vid_score(
    results_dir: Path,
    complete_only: bool,
    video_suffix: str,
    include_text: bool,
    missing_score: float,
) -> dict[str, Any]:
    vid_score: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for path in sorted(results_dir.glob("*.json")):
        if path.name == "agentic_results.json" or path.name.startswith("metrics"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "segments" not in data:
            continue
        if complete_only and data.get("complete") is not True:
            skipped[path.name] = "not complete"
            continue

        video_key = _video_key(str(data.get("video_name") or path.stem), video_suffix)
        segment_scores: dict[str, Any] = {}
        for segment in data.get("segments", []):
            plan = segment.get("plan", {})
            try:
                start = int(plan["start"])
                end = int(plan["end"])
            except (KeyError, TypeError, ValueError):
                continue
            score = segment.get("anomaly_score")
            explanation = str(segment.get("explanation") or "")
            if score is None:
                score, recovered_explanation = parse_reasoning_json(str(segment.get("reasoning_response") or ""))
                if not explanation:
                    explanation = recovered_explanation
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = float(missing_score)
                if not explanation:
                    explanation = str(segment.get("parse_error") or "missing/unparseable score")
            value: list[Any] = [round(score_value, 4), explanation]
            if include_text:
                value.append(str(segment.get("reasoning_response") or ""))
            segment_scores[f"{start}, {end}"] = value

        vid_score[video_key] = segment_scores

    return {
        "config": {
            "source": str(results_dir),
            "complete_only": complete_only,
            "video_suffix": video_suffix,
            "include_text": include_text,
        },
        "vid_score": vid_score,
        "skipped": skipped,
    }


def _video_key(name: str, suffix: str) -> str:
    path = Path(name)
    if path.suffix:
        return path.name
    return f"{path.name}{suffix}"


if __name__ == "__main__":
    main()

