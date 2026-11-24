#!/usr/bin/env python3
"""CIDEr-focused HIVAU semantic evaluation.

CIDEr is a COCO-caption metric. It works best when each item has one generated
answer and one or more references for the same question/task. For HIVAU video
understanding, that means question-level prediction rows are usually more
appropriate than grouped rows that mix detection, event-description, and
reasoning references under one prediction.

Expected prediction row shape:
[
  {
    "video_id": "...",
    "prompt": "...",
    "task_role": "event",
    "pred": "...",
    "refs": ["..."]
  }
]
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path


PTB_PUNCTUATIONS = {
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
}


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    rows = []
    for idx, row in enumerate(data):
        pred = str(row.get("pred") or "").strip()
        refs = [str(ref).strip() for ref in row.get("refs", []) if str(ref).strip()]
        if not pred or not refs:
            continue
        rows.append(
            {
                "id": str(row.get("id") or row.get("video_id") or idx),
                "video_id": str(row.get("video_id") or row.get("id") or idx),
                "prompt": str(row.get("prompt") or "").strip(),
                "task_role": str(row.get("task_role") or row.get("role") or "unknown").strip() or "unknown",
                "pred": pred,
                "refs": refs,
            }
        )
    return rows


def simple_ptb_like_tokenize(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"([{}])".format(re.escape(string.punctuation)), r" \1 ", text)
    tokens = [token for token in text.split() if token not in PTB_PUNCTUATIONS]
    return " ".join(tokens)


def make_coco_inputs(rows, tokenizer):
    gts = {}
    res = {}
    for idx, row in enumerate(rows):
        # Use positional IDs so repeated video_id values from question-level rows
        # cannot overwrite each other.
        item_id = str(idx)
        refs = row["refs"]
        pred = row["pred"]
        if tokenizer == "simple":
            refs = [simple_ptb_like_tokenize(ref) for ref in refs]
            pred = simple_ptb_like_tokenize(pred)
        gts[item_id] = refs
        res[item_id] = [pred]
    return gts, res


def wrap_for_ptb(captions):
    return {key: [{"caption": value} for value in values] for key, values in captions.items()}


def _ptb_worker(gts, res, queue):
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        queue.put({"gts": tokenizer.tokenize(wrap_for_ptb(gts)), "res": tokenizer.tokenize(wrap_for_ptb(res))})
    except Exception as exc:
        queue.put({"error": str(exc)})


def ptb_tokenize_with_timeout(gts, res, timeout):
    queue = mp.Queue()
    process = mp.Process(target=_ptb_worker, args=(gts, res, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise TimeoutError(f"PTBTokenizer timed out after {timeout} seconds")
    if queue.empty():
        raise RuntimeError("PTBTokenizer exited without returning data")
    result = queue.get()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["gts"], result["res"]


def compute_cider(rows, tokenizer="simple", ptb_timeout=120):
    from pycocoevalcap.cider.cider import Cider

    if tokenizer == "ptb":
        raw_gts, raw_res = make_coco_inputs(rows, tokenizer="none")
        gts, res = ptb_tokenize_with_timeout(raw_gts, raw_res, ptb_timeout)
    else:
        gts, res = make_coco_inputs(rows, tokenizer=tokenizer)
    score, item_scores = Cider().compute_score(gts, res)
    return float(score), [float(value) for value in item_scores]


def summarize_rows(rows):
    pred_words = [len(row["pred"].split()) for row in rows]
    ref_words = [sum(len(ref.split()) for ref in row["refs"]) / len(row["refs"]) for row in rows]
    return {
        "num_items": len(rows),
        "avg_pred_words": None if not pred_words else sum(pred_words) / len(pred_words),
        "avg_ref_words": None if not ref_words else sum(ref_words) / len(ref_words),
        "avg_refs_per_item": None if not rows else sum(len(row["refs"]) for row in rows) / len(rows),
    }


def grouped_by_role(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_role"]].append(row)
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description="Compute CIDEr for HIVAU rows with question-aligned reporting.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tokenizer",
        choices=["simple", "none", "ptb"],
        default="simple",
        help="simple is a Windows-safe PTB-like lowercase/punctuation cleanup. ptb uses pycocoevalcap PTBTokenizer.",
    )
    parser.add_argument("--ptb_timeout", type=int, default=120)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Optional task_role filter. Can be passed multiple times, e.g. --role event --role reasoning.",
    )
    parser.add_argument("--by_role", action="store_true", help="Also compute CIDEr separately for each task_role.")
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    if args.role:
        wanted = set(args.role)
        rows = [row for row in rows if row["task_role"] in wanted]
    if not rows:
        raise SystemExit(f"No valid rows found in {args.predictions}")

    score, item_scores = compute_cider(rows, tokenizer=args.tokenizer, ptb_timeout=args.ptb_timeout)
    out = {
        "input": os.path.normpath(args.predictions),
        "protocol": "question_aligned_rows" if any(row["task_role"] != "unknown" for row in rows) else "rows_as_given",
        "tokenizer": args.tokenizer,
        "notes": {
            "CIDEr": "Each row is one generated answer evaluated against references for that same row.",
            "grouped_warning": "If the input groups different HIVAU question types into one refs list, CIDEr will not match a question-aligned protocol.",
            "simple_tokenizer": "Lowercase plus punctuation splitting/removal, used because pycocoevalcap PTBTokenizer can hang on Windows.",
        },
        "summary": summarize_rows(rows),
        "metrics": {"CIDEr": score},
    }

    if args.by_role:
        by_role = {}
        for role, role_rows in sorted(grouped_by_role(rows).items()):
            role_score, _ = compute_cider(role_rows, tokenizer=args.tokenizer, ptb_timeout=args.ptb_timeout)
            by_role[role] = {
                "CIDEr": role_score,
                **summarize_rows(role_rows),
            }
        out["by_role"] = by_role

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
