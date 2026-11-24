#!/usr/bin/env python3
"""Build a model-free manifest for HIVAU video-question inference."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

DEFAULT_INSTRUCTIONS = "data/annotations/hivau_merge_instruction_test_final.jsonl"


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def video_key(path):
    return Path(str(path).replace("\\", "/")).stem


def dataset_key(path):
    return str(path).replace("\\", "/").split("/", 1)[0].lower()


def spread_select(values, count):
    values = sorted({int(value) for value in values})
    if count <= 0 or len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def find_frame(frame_dir, index):
    for extension in (".jpg", ".png", ".jpeg"):
        path = frame_dir / f"{int(index):06d}{extension}"
        if path.exists():
            return path
    return None


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_manifest(args):
    rows = load_jsonl(args.instructions)
    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("type", "")).lower() != "video":
            continue
        key = video_key(row.get("video", ""))
        if not args.video or key.lower() == args.video.lower():
            grouped[(dataset_key(row.get("video", "")), key)].append(row)

    roots = {
        "ucf-crime": (Path(args.ucf_predictions), Path(args.ucf_frames)),
        "xd-violence": (Path(args.xd_predictions), Path(args.xd_frames)),
    }
    manifest, skipped = [], []
    for (dataset, key), questions_in in sorted(grouped.items()):
        if args.max_videos and len(manifest) >= args.max_videos:
            break
        if dataset not in roots:
            skipped.append({"video": key, "reason": f"unsupported_dataset:{dataset}"})
            continue
        prediction_root, frame_root = roots[dataset]
        prediction_path = prediction_root / f"{key}.json"
        frame_dir = frame_root / key
        if not prediction_path.exists() or not frame_dir.is_dir():
            skipped.append({"video": key, "reason": "missing_prediction_or_frames"})
            continue

        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        segments = sorted(prediction.get("segments") or [], key=lambda item: int(item.get("start", 0)))
        contexts, candidates = [], []
        for segment in segments:
            text = str((segment.get("agent_outputs") or {}).get("context_agent") or "").strip()
            if text:
                contexts.append({"start": int(segment["start"]), "end": int(segment["end"]), "text": text})
            candidates.extend(segment.get("mif_reasoning_indices") or [])

        frames, missing = [], []
        for index in spread_select(candidates, args.num_frames):
            frame_path = find_frame(frame_dir, index)
            if frame_path:
                frames.append({"index": index, "path": str(frame_path.resolve())})
            else:
                missing.append(index)
        questions = [
            {
                "id": str(row.get("id")),
                "task": str(row.get("task") or "").strip(),
                "prompt": str(row.get("prompt") or "").strip(),
                "reference": str(row.get("anwser") or row.get("answer") or "").strip(),
            }
            for row in sorted(questions_in, key=lambda item: int(item.get("id", 0)))
        ]
        issues = []
        if not contexts:
            issues.append("no_contexts")
        if not frames:
            issues.append("no_mif_frames")
        if missing:
            issues.append(f"missing_frame_indices:{missing}")
        if not questions:
            issues.append("no_questions")
        if issues and args.strict:
            skipped.append({"video": key, "reason": ";".join(issues)})
            continue
        manifest.append(
            {
                "video_name": key,
                "dataset": dataset,
                "source_video": questions_in[0].get("video"),
                "length": int(prediction.get("length", 0)),
                "prediction_json": str(prediction_path.resolve()),
                "context_segments": contexts,
                "mif_frame_selection": {
                    "policy": "timeline-spread unique saved absolute mif_reasoning_indices",
                    "candidate_count": len({int(value) for value in candidates}),
                    "requested_count": args.num_frames,
                    "indices": [item["index"] for item in frames],
                    "frames": frames,
                },
                "questions": questions,
                "validation_issues": issues,
            }
        )
    return manifest, skipped, len(rows), len(grouped)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--ucf-predictions", default="results/predictions/ucf_crime_mif_stride150")
    parser.add_argument("--xd-predictions", default="results/predictions/xd_violence_mif_stride64")
    parser.add_argument("--ucf-frames", default="data/ucf/frames")
    parser.add_argument("--xd-frames", default="data/xd/frames")
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--video")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="results/vau_hivau/vad_agent_question_conditioned/manifest.jsonl")
    parser.add_argument("--meta-out", default="results/vau_hivau/vad_agent_question_conditioned/manifest_meta.json")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest, skipped, input_rows, candidate_videos = build_manifest(args)
    atomic_write(args.out, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    meta = {
        "instructions": str(Path(args.instructions).resolve()),
        "input_rows": input_rows,
        "candidate_video_count": candidate_videos,
        "manifest_video_count": len(manifest),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "num_frames": args.num_frames,
        "video_filter": args.video,
        "strict": args.strict,
    }
    atomic_write(args.meta_out, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest_videos={len(manifest)} skipped={len(skipped)} out={args.out}")
    if manifest:
        first = manifest[0]
        print(
            f"first_video={first['video_name']} contexts={len(first['context_segments'])} "
            f"frames={len(first['mif_frame_selection']['frames'])} questions={len(first['questions'])}"
        )


if __name__ == "__main__":
    main()



