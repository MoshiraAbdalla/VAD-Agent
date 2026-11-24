#!/usr/bin/env python3
"""Build MIF video predictions aligned to original HIVAU instruction tasks."""

import argparse
import json
import re
from pathlib import Path


def strip_video_extension(video_name):
    lower_name = str(video_name).lower()
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        if lower_name.endswith(ext):
            return str(video_name)[: -len(ext)]
    return str(video_name)


def parse_video_id(video_id):
    parts = str(video_id).replace("\\", "/").split("/")
    return parts[0], strip_video_extension(parts[-1])


def clean_model_text(text):
    text = str(text or "")
    text = re.sub(r"\bcontext\s+summary:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\banswers?\s+to\s+(?:the\s+)?prompt\s+questions?:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsummary:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\boutput\s*:\s*[01]\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\boutput\s*\n\s*[01]\b", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def segment_label(segment):
    try:
        return int(segment.get("label", 0))
    except (TypeError, ValueError):
        return 0


def unique_join(texts):
    out = []
    seen = set()
    for text in texts:
        text = clean_model_text(text)
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return " ".join(out)


def load_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_segments(prediction, max_segments):
    segments = prediction.get("segments") or []
    abnormal = [segment for segment in segments if segment_label(segment) > 0]
    if abnormal:
        return abnormal[: int(max_segments)], "abnormal"
    normal = [segment for segment in segments if segment_label(segment) <= 0]
    return normal[: int(max_segments)], "normal"


def last_summary_segments(prediction, n_segments):
    selected = []
    for segment in reversed(prediction.get("segments") or []):
        outputs = segment.get("agent_outputs") or {}
        if clean_model_text(outputs.get("summarization_agent", "")):
            selected.append(segment)
        if len(selected) >= int(n_segments):
            break
    return list(reversed(selected))


def texts_from_segments(segments, agent_name):
    texts = []
    for segment in segments:
        outputs = segment.get("agent_outputs") or {}
        text = clean_model_text(outputs.get(agent_name, ""))
        if text:
            texts.append(text)
    return unique_join(texts)


def detection_text(selected_mode):
    if selected_mode == "abnormal":
        return "An anomaly exists in the video."
    if selected_mode == "normal":
        return "No anomaly exists in the video."
    return ""


def build_video_prediction_cache(instruction_rows, folders, max_evidence_segments, last_n_segments):
    cache = {}
    missing = []
    video_ids = sorted({str(row.get("video") or "") for row in instruction_rows if row.get("type") == "video"})

    for video_id in video_ids:
        dataset, video_name = parse_video_id(video_id)
        folder = folders.get(dataset)
        pred_path = None if folder is None else folder / f"{video_name}.json"
        prediction = None
        if pred_path and pred_path.exists():
            prediction = load_json(pred_path)
        if prediction is None:
            missing.append({"video": video_id, "prediction_json": "" if pred_path is None else str(pred_path)})
            continue

        evidence_segments, evidence_mode = select_segments(prediction, max_evidence_segments)
        summary_segments = evidence_segments if evidence_mode == "abnormal" else last_summary_segments(prediction, last_n_segments)
        event_pred = texts_from_segments(summary_segments, "summarization_agent")
        reasoning_pred = texts_from_segments(evidence_segments, "reasoning_agent")
        normal_scene_pred = unique_join(
            [
                texts_from_segments(summary_segments, "summarization_agent"),
                texts_from_segments(evidence_segments, "context_agent"),
            ]
        )

        cache[video_id] = {
            "prediction_json": "" if pred_path is None else str(pred_path),
            "evidence_mode": evidence_mode,
            "judgement": detection_text(evidence_mode),
            "description": event_pred if evidence_mode == "abnormal" else normal_scene_pred,
            "analysis": reasoning_pred,
            "source_segments": [
                {
                    "start": int(segment.get("start", 0)),
                    "end": int(segment.get("end", 0)),
                    "label": segment_label(segment),
                }
                for segment in evidence_segments
            ],
        }

    return cache, missing


def build_rows(instruction_rows, folders, max_evidence_segments, last_n_segments):
    video_rows = [row for row in instruction_rows if row.get("type") == "video"]
    cache, missing = build_video_prediction_cache(video_rows, folders, max_evidence_segments, last_n_segments)
    out = []

    for row in video_rows:
        instruction_id = str(row.get("id"))
        video = str(row.get("video") or "")
        task = str(row.get("task") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        ref = str(row.get("anwser") or row.get("answer") or "").strip()
        prediction_info = cache.get(video)
        if not prediction_info:
            continue
        pred = clean_model_text(prediction_info.get(task, ""))
        if not pred or not ref:
            missing.append({"id": instruction_id, "video": video, "task": task, "reason": "empty_pred_or_ref"})
            continue

        out.append(
            {
                "id": instruction_id,
                "video_id": f"{video}#instruction_{instruction_id}",
                "source_video_id": video,
                "hivau_instruction_id": instruction_id,
                "hivau_type": row.get("type"),
                "task": task,
                "task_role": task,
                "prompt": prompt,
                "pred": pred,
                "refs": [ref],
                "source": {
                    "prediction_json": prediction_info["prediction_json"],
                    "evidence_mode": prediction_info["evidence_mode"],
                    "evidence_segments": prediction_info["source_segments"],
                },
            }
        )

    return out, missing


def main():
    parser = argparse.ArgumentParser(description="Build MIF rows with original HIVAU video task names.")
    parser.add_argument(
        "--instructions",
        default="data/annotations/hivau_merge_instruction_test_final.jsonl",
    )
    parser.add_argument("--ucf-score-folder", default="results/predictions/ucf_crime_mif_stride150")
    parser.add_argument("--xd-score-folder", default="results/predictions/xd_violence_mif_stride150")
    parser.add_argument("--max-evidence-segments", type=int, default=2)
    parser.add_argument("--last-n-segments", type=int, default=2)
    parser.add_argument("--out", default="results/vau_hivau/vad_agent/mif_hivau_video_original_task_predictions.json")
    parser.add_argument("--meta-out", default="results/vau_hivau/vad_agent/mif_hivau_video_original_task_predictions_meta.json")
    args = parser.parse_args()

    folders = {
        "ucf-crime": Path(args.ucf_score_folder),
        "xd-violence": Path(args.xd_score_folder),
    }
    instruction_rows = load_jsonl(args.instructions)
    out, missing = build_rows(
        instruction_rows,
        folders=folders,
        max_evidence_segments=args.max_evidence_segments,
        last_n_segments=args.last_n_segments,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    task_counts = {}
    for row in out:
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1

    meta = {
        "instructions": str(Path(args.instructions)),
        "ucf_score_folder": str(Path(args.ucf_score_folder)),
        "xd_score_folder": str(Path(args.xd_score_folder)),
        "input_instruction_rows": len(instruction_rows),
        "input_video_instruction_rows": sum(1 for row in instruction_rows if row.get("type") == "video"),
        "output_rows": len(out),
        "task_counts": task_counts,
        "missing_count": len(missing),
        "missing": missing,
        "mapping": {
            "judgement": "Derived anomaly/no-anomaly answer from MIF selected evidence segments.",
            "description": "MIF summarization_agent from selected abnormal segments; normal videos use last summaries plus context.",
            "analysis": "MIF reasoning_agent from selected evidence segments.",
        },
        "id_policy": "id is the original HIVAU instruction id; video_id appends #instruction_<id> to keep metric item ids unique.",
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"input_instruction_rows={len(instruction_rows)}")
    print(f"input_video_instruction_rows={meta['input_video_instruction_rows']}")
    print(f"output_rows={len(out)}")
    print(f"task_counts={task_counts}")
    print(f"missing_count={len(missing)}")
    print(f"out={out_path}")
    print(f"meta_out={args.meta_out}")


if __name__ == "__main__":
    main()




