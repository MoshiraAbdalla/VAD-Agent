#!/usr/bin/env python3
"""
Compute text-generation metrics for saved HolmesVAU/HIVAU prediction JSON files.

The input is the JSON produced by evaluation/hivau_task_video_evaluation.py:
[
  {
    "video_id": "...",
    "refs": ["reference 1", "reference 2", ...],
    "pred": "model output"
  }
]

This script computes the caption metrics used in the Holmes-VAU paper tables
when the local dependencies are available:
  - BLEU-1..4. The BLEU field stores the cumulative BLEU-1 through BLEU-4 values.
  - METEOR
  - ROUGE-L, reported as ROUGE

It can also compute BeScore as BERTScore F1 if the optional bert_score package
and model weights are installed/cached. It can compute PaScore with the
installed parascore package, but exact paper reproduction depends on the
authors' original ParaScore configuration.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

# Avoid repeated Hugging Face network HEAD retries once the model is cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from bert_score import score as bert_score_score
except Exception as exc:
    bert_score_score = None
    BERT_SCORE_IMPORT_ERROR = exc
else:
    BERT_SCORE_IMPORT_ERROR = None

try:
    from parascore import ParaScorer
except Exception as exc:
    ParaScorer = None
    PARASCORE_IMPORT_ERROR = exc
else:
    PARASCORE_IMPORT_ERROR = None


def load_prediction_rows(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = [rows]

    items = []
    seen_ids = set()
    for i, row in enumerate(rows):
        pred = (row.get("pred") or "").strip()
        refs = [str(ref).strip() for ref in row.get("refs", []) if str(ref).strip()]
        item_id = str(row.get("id") or row.get("video_id") or i)
        if item_id in seen_ids:
            item_id = f"{item_id}__row_{i}"
        seen_ids.add(item_id)
        prompt = (row.get("prompt") or "").strip()
        if pred and refs:
            items.append({"id": item_id, "pred": pred, "refs": refs, "prompt": prompt})
    return items


def make_pycoco_inputs(items):
    gts = {item["id"]: item["refs"] for item in items}
    # pycocoevalcap scorers expect result values to be lists.
    res = {item["id"]: [item["pred"]] for item in items}
    return gts, res


def normalize_meteor_text(text, max_words=None):
    words = str(text).replace("|||", " ").split()
    if max_words is not None and max_words > 0:
        words = words[:max_words]
    return " ".join(words)


def normalize_meteor_inputs(gts, res, max_words=None):
    clean_gts = {
        item_id: [normalize_meteor_text(ref, max_words=max_words) for ref in refs]
        for item_id, refs in gts.items()
    }
    clean_res = {
        item_id: [normalize_meteor_text(pred, max_words=max_words) for pred in preds]
        for item_id, preds in res.items()
    }
    return clean_gts, clean_res


def _meteor_worker(gts, res, queue, max_words=None):
    try:
        from pycocoevalcap.meteor.meteor import Meteor

        gts, res = normalize_meteor_inputs(gts, res, max_words=max_words)
        score, _ = Meteor().compute_score(gts, res)
    except Exception as exc:
        queue.put({"error": str(exc)})
    else:
        queue.put({"score": float(score)})


def compute_meteor_with_timeout(gts, res, timeout_seconds, max_words=None):
    queue = mp.Queue()
    process = mp.Process(target=_meteor_worker, args=(gts, res, queue, max_words))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, f"METEOR timed out after {timeout_seconds} seconds."

    if queue.empty():
        return None, "METEOR process exited without returning a score."

    result = queue.get()
    if "error" in result:
        return None, result["error"]
    return result["score"], None


def compute_meteor_per_item(items, timeout_seconds, max_words=None):
    scores = []
    errors = []
    timings = {}

    for index, item in enumerate(items, start=1):
        start = time.perf_counter()
        gts = {item["id"]: item["refs"]}
        res = {item["id"]: [item["pred"]]}
        score, error = compute_meteor_with_timeout(gts, res, timeout_seconds, max_words=max_words)
        timings[item["id"]] = time.perf_counter() - start
        if error:
            errors.append({"id": item["id"], "error": error, "seconds": round(timings[item["id"]], 3)})
        else:
            scores.append(float(score))
        if index % 10 == 0:
            print(f"METEOR per-item progress: {index}/{len(items)}", flush=True)

    if not scores:
        return None, errors, timings
    return sum(scores) / len(scores), errors, timings


def compute_caption_metrics(
    items,
    include_meteor=True,
    meteor_timeout=300,
    meteor_per_item=False,
    meteor_max_words=None,
):
    gts, res = make_pycoco_inputs(items)
    metrics = {}
    skipped = {}
    timings = {}

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
    except Exception as exc:
        return metrics, {"pycocoevalcap": str(exc)}

    scorers = [
        ("BLEU", Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        ("ROUGE", Rouge(), "ROUGE"),
    ]

    if include_meteor:
        if shutil.which("java") is None:
            skipped["METEOR"] = "Java was not found on PATH; pycocoevalcap METEOR requires Java."
        else:
            start = time.perf_counter()
            if meteor_per_item:
                meteor_score, meteor_error, meteor_item_timings = compute_meteor_per_item(
                    items,
                    meteor_timeout,
                    max_words=meteor_max_words,
                )
                timings["METEOR_per_item_slowest"] = dict(
                    sorted(meteor_item_timings.items(), key=lambda item: item[1], reverse=True)[:20]
                )
            else:
                meteor_score, meteor_error = compute_meteor_with_timeout(
                    gts,
                    res,
                    meteor_timeout,
                    max_words=meteor_max_words,
                )
            timings["METEOR"] = time.perf_counter() - start
            if meteor_score is not None:
                metrics["METEOR"] = float(meteor_score)
            if meteor_error:
                skipped["METEOR"] = meteor_error

    for label, scorer, method in scorers:
        try:
            start = time.perf_counter()
            score, _ = scorer.compute_score(gts, res)
            timings[label] = time.perf_counter() - start
        except Exception as exc:
            skipped[label] = str(exc)
            continue

        if isinstance(method, list):
            for name, value in zip(method, score):
                metrics[name] = float(value)
            metrics["BLEU"] = [metrics[name] for name in method]
            metrics["BLEU_acc"] = float(sum(metrics[name] for name in method))
        else:
            metrics[method] = float(score)

    return metrics, skipped, timings


def compute_bertscore(items, model_type=None, lang="en", rescale_with_baseline=False):
    if bert_score_score is None:
        return None, f"bert_score is not installed or could not be imported: {BERT_SCORE_IMPORT_ERROR}"

    candidates = []
    references = []
    owner_ids = []
    for item in items:
        for ref in item["refs"]:
            candidates.append(item["pred"])
            references.append(ref)
            owner_ids.append(item["id"])

    try:
        _, _, f1 = bert_score_score(
            candidates,
            references,
            lang=lang,
            model_type=model_type,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
    except Exception as exc:
        return None, str(exc)

    best_by_id = {}
    for item_id, score_value in zip(owner_ids, f1.tolist()):
        best_by_id[item_id] = max(best_by_id.get(item_id, float("-inf")), float(score_value))

    if not best_by_id:
        return None, "no BERTScore pairs were computed"
    return sum(best_by_id.values()) / len(best_by_id), None


def compute_parascore(
    items,
    model_type=None,
    lang="en",
    mode="similarity",
    rescale_with_baseline=False,
):
    if ParaScorer is None:
        return None, f"parascore is not installed or could not be imported: {PARASCORE_IMPORT_ERROR}"

    if mode not in {"similarity", "base", "free"}:
        return None, f"unsupported ParaScore mode: {mode}"

    candidates = [item["pred"] for item in items]
    references = [item["refs"] for item in items]

    try:
        scorer = ParaScorer(
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
        )
        if mode == "similarity":
            _, _, f1 = scorer.score(candidates, references)
            values = f1.tolist()
        else:
            sources = [item.get("prompt") or item["refs"][0] for item in items]
            if mode == "free":
                values = scorer.free_score(candidates, sources)
            else:
                # The package base_score requires a source, candidate, and refs.
                # Our video-level prediction JSON has no original paraphrase source,
                # so this falls back to prompt when present and the first ref otherwise.
                first_refs = [refs[0] for refs in references]
                values = scorer.base_score(candidates, sources, first_refs)
    except Exception as exc:
        return None, str(exc)

    if not values:
        return None, "no ParaScore values were computed"
    return sum(float(value) for value in values) / len(values), None


def load_external_parascore(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "PaScore" in data:
            return float(data["PaScore"]), None
        if "ParaScore" in data:
            return float(data["ParaScore"]), None
        values = [float(v) for v in data.values() if isinstance(v, (int, float))]
        if values:
            return sum(values) / len(values), None

    if isinstance(data, list):
        values = []
        for row in data:
            if isinstance(row, dict):
                value = row.get("PaScore", row.get("ParaScore", row.get("score")))
                if isinstance(value, (int, float)):
                    values.append(float(value))
        if values:
            return sum(values) / len(values), None

    return None, "unrecognized ParaScore JSON format"


def round_nested(value):
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value


def main():
    parser = argparse.ArgumentParser(description="Compute HIVAU semantic metrics from saved prediction JSON.")
    parser.add_argument("--predictions", default="evaluation/hivau_video_predictions.json")
    parser.add_argument("--out", default="evaluation/hivau_video_metrics_available.json")
    parser.add_argument("--skip_meteor", action="store_true")
    parser.add_argument("--meteor_timeout", type=int, default=300, help="Maximum seconds to wait for METEOR.")
    parser.add_argument("--meteor_per_item", action="store_true", help="Compute METEOR one id at a time and report slow or failing ids.")
    parser.add_argument(
        "--meteor_max_words",
        type=int,
        default=None,
        help="Optional METEOR-only cap on prediction/reference words. Default is exact/no cap.",
    )
    parser.add_argument("--bertscore", action="store_true", help="Compute BeScore as BERTScore F1 if available.")
    parser.add_argument("--bertscore_model", default=None, help="Optional BERTScore model_type, e.g. roberta-large.")
    parser.add_argument("--bertscore_lang", default="en")
    parser.add_argument("--bertscore_rescale", action="store_true")
    parser.add_argument("--parascore", action="store_true", help="Compute PaScore using the installed parascore package if available.")
    parser.add_argument("--parascore_model", default=None, help="Optional ParaScore model_type; defaults to the package language default.")
    parser.add_argument("--parascore_lang", default="en")
    parser.add_argument("--parascore_mode", choices=["similarity", "base", "free"], default="similarity")
    parser.add_argument("--parascore_rescale", action="store_true")
    parser.add_argument("--parascore_json", help="Optional external ParaScore JSON to merge into the output.")
    args = parser.parse_args()

    items = load_prediction_rows(args.predictions)
    if not items:
        print("No valid prediction rows found:", args.predictions)
        sys.exit(1)

    total_start = time.perf_counter()
    metrics, skipped, timings = compute_caption_metrics(
        items,
        include_meteor=not args.skip_meteor,
        meteor_timeout=args.meteor_timeout,
        meteor_per_item=args.meteor_per_item,
        meteor_max_words=args.meteor_max_words,
    )

    if args.bertscore:
        start = time.perf_counter()
        bescore, error = compute_bertscore(
            items,
            model_type=args.bertscore_model,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale,
        )
        timings["BeScore"] = time.perf_counter() - start
        if error:
            skipped["BeScore"] = error
        else:
            metrics["BeScore"] = float(bescore)

    if args.parascore:
        start = time.perf_counter()
        pascore, error = compute_parascore(
            items,
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
    elif args.parascore_json:
        start = time.perf_counter()
        pascore, error = load_external_parascore(args.parascore_json)
        timings["PaScore"] = time.perf_counter() - start
        if error:
            skipped["PaScore"] = error
        else:
            metrics["PaScore"] = float(pascore)
    else:
        skipped["PaScore"] = "ParaScore implementation is not included in this repository."

    out = {
        "input": os.path.normpath(args.predictions),
        "num_items": len(items),
        "multi_reference_policy": {
            "caption_metrics": "All references are passed to pycocoevalcap.",
            "BeScore": "If enabled, compute BERTScore F1 for each prediction/reference pair and average the best reference score per item.",
            "PaScore": "If --parascore is enabled, compute with the installed parascore package. Default mode is prediction/reference similarity because this prediction JSON has no separate paraphrase source text.",
            "BLEU": "BLEU stores cumulative BLEU-1, BLEU-2, BLEU-3, and BLEU-4 values in order.",
            "BLEU_acc": "Accumulated BLEU requested for reporting: BLEU-1 + BLEU-2 + BLEU-3 + BLEU-4.",
            "METEOR": "Default mode computes corpus METEOR. With --meteor_per_item, each video id is scored separately and successful scores are averaged.",
        },
        "meteor_settings": {
            "timeout_seconds": args.meteor_timeout,
            "per_item": bool(args.meteor_per_item),
            "max_words": args.meteor_max_words,
            "exact": args.meteor_max_words is None,
        },
        "metrics": metrics,
        "skipped": skipped,
        "timings_seconds": round_nested(timings),
        "total_seconds": round(time.perf_counter() - total_start, 3),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
