#!/usr/bin/env python3
"""Compute question-aligned HIVAU semantic metrics and table output.

This is the single entry point for question-aligned HIVAU prediction files.
Each prediction row is treated as one original HIVAU question answer. If the
instruction JSONL is supplied, task labels are copied from the GT `task` field
so the per-task split uses the dataset names, e.g. judgement, description, and
analysis.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from compute_hivau_cider_aligned import compute_cider, summarize_rows
from compute_hivau_metrics import (
    compute_bertscore,
    compute_caption_metrics,
    compute_parascore,
    load_external_parascore,
    round_nested,
)


TABLE_COLUMNS = [
    ("BLEU", "BLEU_acc"),
    ("CIDEr", "CIDEr"),
    ("METEOR", "METEOR"),
    ("ROUGE", "ROUGE"),
    ("BERTScore", "BeScore"),
    ("PaScore", "PaScore"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def normalize_id(value):
    return str(value).strip()


def conversation_value(row, speaker):
    for turn in row.get("conversations") or []:
        if str(turn.get("from") or "").lower() == speaker:
            return str(turn.get("value") or "").strip()
    return ""


def prompt_from_instruction(row):
    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        return prompt
    prompt = conversation_value(row, "human")
    if prompt.startswith("<video>"):
        prompt = prompt[len("<video>") :].strip()
    return prompt


def answer_from_instruction(row):
    answer = str(row.get("answer") or row.get("anwser") or "").strip()
    if answer:
        return answer
    return conversation_value(row, "gpt")


def instruction_index(path):
    if not path:
        return {}
    by_id = {}
    for row in load_jsonl(path):
        row_id = normalize_id(row.get("id"))
        if row_id:
            by_id[row_id] = row
    return by_id


def dataset_from_video_path(value):
    video = str(value or "").strip().replace("\\", "/").lower()
    if video.startswith("ucf-crime/"):
        return "ucf"
    if video.startswith("xd-violence/"):
        return "xd"
    return "unknown"


def clean_refs(refs):
    return [str(ref).strip() for ref in refs or [] if str(ref).strip()]


def prepare_rows(prediction_path, instructions_path=None, synced_predictions_out=None, dataset_filter="all"):
    raw = load_json(prediction_path)
    if isinstance(raw, dict):
        raw = [raw]

    dataset_filter = str(dataset_filter or "all").lower()
    gt_by_id = instruction_index(instructions_path)
    rows = []
    synced_rows = []
    skipped = []
    sync_report = {
        "instructions": os.path.normpath(instructions_path) if instructions_path else None,
        "dataset_filter": dataset_filter,
        "dataset_counts": defaultdict(int),
        "filtered_out_rows": 0,
        "changed_task_role": 0,
        "missing_instruction_ids": [],
        "mismatches": [],
    }

    for idx, row in enumerate(raw):
        out_row = dict(row)
        row_id = normalize_id(row.get("id") if row.get("id") is not None else idx)
        instruction = gt_by_id.get(row_id)
        instruction_video = instruction.get("video") if instruction else None
        row_dataset = dataset_from_video_path(instruction_video or row.get("video_id") or row.get("video"))
        sync_report["dataset_counts"][row_dataset] += 1
        if dataset_filter != "all" and row_dataset != dataset_filter:
            sync_report["filtered_out_rows"] += 1
            continue

        if instruction:
            dataset_task = str(instruction.get("task") or "").strip()
            if dataset_task:
                old_role = str(out_row.get("task_role") or "").strip()
                if old_role and old_role != dataset_task:
                    out_row.setdefault("inferred_task_role", old_role)
                    sync_report["changed_task_role"] += 1
                out_row["task_role"] = dataset_task
                out_row["task"] = dataset_task
            out_row["hivau_type"] = instruction.get("type")
            if not clean_refs(out_row.get("refs")):
                instruction_answer = answer_from_instruction(instruction)
                if instruction_answer:
                    out_row["refs"] = [instruction_answer]

            prediction_video = row.get("source_video_id") or row.get("video") or row.get("video_id")
            if isinstance(prediction_video, str) and "#instruction_" in prediction_video:
                prediction_video = prediction_video.split("#instruction_", 1)[0]
            if prediction_video and instruction.get("video") and prediction_video != instruction.get("video"):
                sync_report["mismatches"].append(
                    {
                        "id": row_id,
                        "field": "video",
                        "prediction": prediction_video,
                        "instruction": instruction.get("video"),
                    }
                )
            instruction_prompt = prompt_from_instruction(instruction)
            if row.get("prompt") and instruction_prompt and row.get("prompt") != instruction_prompt:
                sync_report["mismatches"].append({"id": row_id, "field": "prompt"})
        elif instructions_path:
            sync_report["missing_instruction_ids"].append(row_id)

        pred = str(out_row.get("pred") or "").strip()
        refs = clean_refs(out_row.get("refs"))
        if not pred or not refs:
            skipped.append({"row_index": idx, "id": row_id, "reason": "missing pred or refs"})
            synced_rows.append(out_row)
            continue

        task = str(out_row.get("task_role") or out_row.get("task") or out_row.get("role") or "unknown").strip()
        if not task:
            task = "unknown"

        metric_id = f"row_{idx}_{row_id}"
        rows.append(
            {
                "metric_id": metric_id,
                "id": row_id,
                "video_id": str(out_row.get("video_id") or row_id),
                "prompt": str(out_row.get("prompt") or "").strip(),
                "task_role": task,
                "pred": pred,
                "refs": refs,
            }
        )
        synced_rows.append(out_row)

    if synced_predictions_out:
        save_json(synced_predictions_out, synced_rows)

    sync_report["num_input_rows"] = len(raw)
    sync_report["num_metric_rows"] = len(rows)
    sync_report["num_skipped_rows"] = len(skipped)
    sync_report["skipped_rows"] = skipped[:50]
    sync_report["num_mismatches"] = len(sync_report["mismatches"])
    sync_report["mismatches"] = sync_report["mismatches"][:50]
    sync_report["num_missing_instruction_ids"] = len(sync_report["missing_instruction_ids"])
    sync_report["missing_instruction_ids"] = sync_report["missing_instruction_ids"][:50]
    sync_report["dataset_counts"] = dict(sorted(sync_report["dataset_counts"].items()))
    return rows, sync_report


def rows_to_caption_items(rows):
    return [
        {
            "id": row["metric_id"],
            "pred": row["pred"],
            "refs": row["refs"],
            "prompt": row.get("prompt") or "",
        }
        for row in rows
    ]


def rows_to_cider_rows(rows):
    return [
        {
            "id": row["metric_id"],
            "video_id": row["video_id"],
            "prompt": row.get("prompt") or "",
            "task_role": row["task_role"],
            "pred": row["pred"],
            "refs": row["refs"],
        }
        for row in rows
    ]


def compute_metric_block(rows, args, label):
    caption_items = rows_to_caption_items(rows)
    cider_rows = rows_to_cider_rows(rows)

    block_start = time.perf_counter()
    metrics, skipped, timings = compute_caption_metrics(
        caption_items,
        include_meteor=not args.skip_meteor,
        meteor_timeout=args.meteor_timeout,
        meteor_per_item=args.meteor_per_item,
        meteor_max_words=args.meteor_max_words,
    )

    cider_score, _ = compute_cider(cider_rows, tokenizer=args.cider_tokenizer, ptb_timeout=args.ptb_timeout)
    metrics["CIDEr"] = float(cider_score)

    if args.bertscore:
        start = time.perf_counter()
        bescore, error = compute_bertscore(
            caption_items,
            model_type=args.bertscore_model,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale,
        )
        timings["BeScore"] = time.perf_counter() - start
        if error:
            skipped["BeScore"] = error
        else:
            metrics["BeScore"] = float(bescore)
            metrics["BERTScore"] = float(bescore)

    if args.parascore:
        start = time.perf_counter()
        pascore, error = compute_parascore(
            caption_items,
            model_type=args.parascore_model,
            lang=args.parascore_lang,
            mode=args.parascore_mode,
            rescale_with_baseline=args.parascore_rescale,
        )
        timings["PaScore"] = time.perf_counter() - start
        if error:
            skipped["PaScore"] = error
        else:
            metrics["PaScore"] = float(pascore)
    elif args.parascore_json and label == "overall":
        start = time.perf_counter()
        pascore, error = load_external_parascore(args.parascore_json)
        timings["PaScore"] = time.perf_counter() - start
        if error:
            skipped["PaScore"] = error
        else:
            metrics["PaScore"] = float(pascore)

    if "BLEU_acc" not in metrics and all(f"BLEU-{idx}" in metrics for idx in range(1, 5)):
        metrics["BLEU_acc"] = float(sum(metrics[f"BLEU-{idx}"] for idx in range(1, 5)))

    return {
        "num_items": len(rows),
        "summary": summarize_rows(cider_rows),
        "metrics": metrics,
        "skipped": skipped,
        "timings_seconds": round_nested(timings),
        "total_seconds": round(time.perf_counter() - block_start, 3),
    }


def group_by_task(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_role"]].append(row)
    return dict(sorted(groups.items()))


def metric_value(metrics, key):
    value = metrics.get(key)
    if value is None and key == "BeScore":
        value = metrics.get("BERTScore")
    return value


def format_float(value, digits=3):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def table_rows(overall, by_task, digits=3):
    rows = []
    for split, block in [("overall", overall)] + [(task, by_task[task]) for task in sorted(by_task)]:
        metrics = block.get("metrics", {})
        row = {"split": split, "num_items": block.get("num_items")}
        for header, key in TABLE_COLUMNS:
            row[header] = format_float(metric_value(metrics, key), digits=digits)
        rows.append(row)
    return rows


def write_table_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "num_items"] + [header for header, _ in TABLE_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    headers = ["split", "num_items"] + [header for header, _ in TABLE_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def default_sidecar(out_path, suffix):
    path = Path(out_path)
    return str(path.with_suffix(suffix))


def main():
    parser = argparse.ArgumentParser(description="Compute all question-aligned HIVAU semantic metrics in one command.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--instructions", default="HIVAU-70k/instruction/merge_instruction_test_final.jsonl")
    parser.add_argument("--out", required=True, help="Output JSON report.")
    parser.add_argument("--table_csv", default=None, help="CSV table output. Defaults to OUT with .csv suffix.")
    parser.add_argument("--table_md", default=None, help="Markdown table output. Defaults to OUT with .md suffix.")
    parser.add_argument("--synced_predictions_out", default=None, help="Optional prediction JSON with GT task labels copied in.")
    parser.add_argument("--dataset_filter", choices=["all", "ucf", "xd"], default="all", help="Restrict metric rows by GT HIVAU video path prefix.")
    parser.add_argument("--no_by_task", action="store_true", help="Only compute the overall table row.")
    parser.add_argument("--table_digits", type=int, default=3)
    parser.add_argument("--skip_meteor", action="store_true")
    parser.add_argument("--meteor_timeout", type=int, default=300)
    parser.add_argument("--meteor_per_item", action="store_true")
    parser.add_argument("--meteor_max_words", type=int, default=None)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore_model", default=None)
    parser.add_argument("--bertscore_lang", default="en")
    parser.add_argument("--bertscore_rescale", action="store_true")
    parser.add_argument("--parascore", action="store_true")
    parser.add_argument("--parascore_model", default=None)
    parser.add_argument("--parascore_lang", default="en")
    parser.add_argument("--parascore_mode", choices=["similarity", "base", "free"], default="similarity")
    parser.add_argument("--parascore_rescale", action="store_true")
    parser.add_argument("--parascore_json", default=None)
    parser.add_argument("--cider_tokenizer", choices=["simple", "none", "ptb"], default="simple")
    parser.add_argument("--ptb_timeout", type=int, default=120)
    args = parser.parse_args()

    rows, sync_report = prepare_rows(
        args.predictions,
        instructions_path=args.instructions,
        synced_predictions_out=args.synced_predictions_out,
        dataset_filter=args.dataset_filter,
    )
    if not rows:
        raise SystemExit(f"No valid prediction rows found in {args.predictions}")

    total_start = time.perf_counter()
    overall = compute_metric_block(rows, args, label="overall")

    by_task = {}
    if not args.no_by_task:
        for task, task_rows in group_by_task(rows).items():
            print(f"Computing task split: {task} ({len(task_rows)} rows)", flush=True)
            by_task[task] = compute_metric_block(task_rows, args, label=task)

    table = table_rows(overall, by_task, digits=args.table_digits)
    table_md = args.table_md or default_sidecar(args.out, ".md")
    table_csv = args.table_csv or default_sidecar(args.out, ".csv")
    write_table_csv(table_csv, table)
    write_text(table_md, markdown_table(table))

    out = {
        "input": os.path.normpath(args.predictions),
        "instructions": os.path.normpath(args.instructions) if args.instructions else None,
        "protocol": "question_aligned_rows",
        "dataset_filter": args.dataset_filter,
        "num_items": len(rows),
        "metrics": overall["metrics"],
        "overall": overall,
        "by_task": by_task,
        "table": table,
        "outputs": {
            "json": os.path.normpath(args.out),
            "table_csv": os.path.normpath(table_csv),
            "table_md": os.path.normpath(table_md),
            "synced_predictions": os.path.normpath(args.synced_predictions_out) if args.synced_predictions_out else None,
        },
        "sync_report": sync_report,
        "notes": {
            "alignment": "Each row is one original HIVAU question; metric IDs are unique per row, so repeated video_id values cannot overwrite rows.",
            "task_labels": "When --instructions is supplied, task_role/task are copied from the HIVAU GT task field.",
            "BLEU": "The table BLEU column is BLEU_acc = BLEU-1 + BLEU-2 + BLEU-3 + BLEU-4.",
            "CIDEr": "CIDEr is computed on the same question-aligned rows, not grouped per-video refs.",
            "BERTScore": "The JSON keeps the existing BeScore key and also adds BERTScore as an alias when computed.",
        },
        "settings": {
            "meteor_timeout": args.meteor_timeout,
            "meteor_per_item": bool(args.meteor_per_item),
            "meteor_max_words": args.meteor_max_words,
            "bertscore": bool(args.bertscore),
            "bertscore_model": args.bertscore_model,
            "parascore": bool(args.parascore),
            "parascore_model": args.parascore_model,
            "parascore_mode": args.parascore_mode,
            "cider_tokenizer": args.cider_tokenizer,
            "dataset_filter": args.dataset_filter,
        },
        "total_seconds": round(time.perf_counter() - total_start, 3),
    }
    save_json(args.out, out)

    print(markdown_table(table))
    print(json.dumps({"out": args.out, "table_csv": table_csv, "table_md": table_md}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
