#!/usr/bin/env python3
"""Run resumable Llama-narrative plus InternVL HIVAU question inference."""

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

NARRATIVE_SYSTEM = (
    "You are a context summarization agent.\n\n"
    "Your task is to summarize the segment contexts into one updated story memory.\n"
    "Write it as a concise temporal story. For example: \"First, a person was standing in a parking "
    "lot. Then, the person was seen running away. Finally, the person disappeared into the crowd.\"\n"
    "Use this only as a style example; do not copy details from it.\n\n"
    "Do not infer hidden intent or add details not present in the text.\n"
    "Keep the answer concise.\n"
    "After the story, provide:\n"
    "Key Entities: <the important people, animals, vehicles, and objects>\n"
    "Scene Settings: <the important locations and environments>\n"
    "Do not start your answer with :  Sure! Here is the summary of the segment contexts \n"
    "Answer only in this format, and do not add any other text or commentary.\n"
    "Summary: <your story summary>\n   Key Entities: <your list of key entities>\n   Scene Settings: <your list of scene settings>\n"
)

ANOMALY_TYPES = (
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Car Accident",
    "Explosion",
    "Fighting",
    "Riot",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
)

TASK_OUTPUT_INSTRUCTIONS = {
    "judgement": (
        "Answer in exactly one sentence:\n"
        "\"No anomaly exists in the video.\"\n"
        "or\n"
        "\"An anomaly exists, specifically <event type>.\"\n"
        "If an anomaly exists, replace <event type> with exactly one label from this closed "
        "dataset vocabulary: "
        + ", ".join(ANOMALY_TYPES)
        + ". Do not invent labels, combine labels, or provide explanations."
    ),
    "description": (
        "Describe only the principal anomaly event in 1-3 concise sentences. "
        "If no anomaly exists, briefly describe the normal scene."
    ),
    "analysis": (
        "Explain in no more than two sentences why the identified event is anomalous. "
        "Do not repeat the complete video summary."
    ),
}


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def chat_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"


def load_llama(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_model_path, trust_remote_code=True, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.llama_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_model_path,
        torch_dtype=dtype,
        device_map=args.llama_device_map,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    ).eval()
    return model, tokenizer


def non_overlapping_contexts(context_segments):
    selected = []
    next_start = 0
    for item in sorted(context_segments, key=lambda value: int(value["start"])):
        start = int(item["start"])
        end = int(item["end"])
        if not selected or start >= next_start:
            selected.append(item)
            next_start = end
    return selected


def generate_narrative(model, tokenizer, record, args):
    selected_contexts = non_overlapping_contexts(record["context_segments"])
    observations = "\n\n".join(
        f"Frames [{item['start']}, {item['end']}): {item['text']}"
        for item in selected_contexts
    )
    messages = [
        {"role": "system", "content": NARRATIVE_SYSTEM},
        {"role": "user", "content": "Create one chronological narrative:\n\n" + observations},
    ]
    prompt = chat_prompt(tokenizer, messages)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=args.llama_max_input_tokens
    )
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.llama_max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def image_transform(size=448):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_internvl(args):
    model = AutoModel.from_pretrained(
        args.internvl_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        cache_dir=args.cache_dir,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.internvl_model_path, trust_remote_code=True, cache_dir=args.cache_dir
    )
    return model, tokenizer


def load_holmes(args):
    model = AutoModel.from_pretrained(
        args.holmes_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        local_files_only=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.holmes_model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    return model, tokenizer


def load_pixels(record, transform):
    tensors = []
    for item in record["mif_frame_selection"]["frames"]:
        with Image.open(item["path"]) as image:
            tensors.append(transform(image))
    if not tensors:
        raise ValueError(f"No frames available for {record['video_name']}")
    return torch.stack(tensors).to(dtype=torch.bfloat16, device="cuda", non_blocking=True)


def answer_question(model, tokenizer, pixels, narrative, question, args):
    task = str(question.get("task") or "").strip().lower()
    output_instruction = TASK_OUTPUT_INSTRUCTIONS.get(
        task, "Answer the question directly and concisely."
    )
    frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(pixels)))
    prompt = (
        frame_prefix
        + "These are chronological frames from one video. The full summary of the video is:\n"
        + narrative
        + "\n\nAnswer the question using both the frames and narrative. Be direct and specific.\n"
        + "Output requirements:\n"
        + output_instruction
        + "\n"
        + "Question: "
        + question["prompt"]
    )
    prompt_holmes = (
        frame_prefix
        + "These are chronological frames from one video. The full summary of the video is:\n"
        + narrative
        + "\n\nAnswer the question using both the frames and narrative. Be direct and specific.\n"
        # + "Output requirements:\n"
        # + output_instruction
        # + "\n"
        + "Question: "
        + question["prompt"]
    )
    max_new_tokens = {
        "judgement": args.judgement_max_new_tokens,
        "description": args.description_max_new_tokens,
        "analysis": args.analysis_max_new_tokens,
    }.get(task, args.internvl_max_new_tokens)
    generation_config = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": (
            args.holmes_max_new_tokens
            if args.answer_backend == "holmes"
            else max_new_tokens
        ),
    }
    with torch.inference_mode():
        if args.answer_backend == "holmes":
            response, _history = model.chat(
                tokenizer,
                pixels,
                prompt_holmes,
                generation_config,
                num_patches_list=[1] * len(pixels),
                history=None,
                return_history=True,
            )
            return response.strip()
        responses = model.batch_chat(
            tokenizer,
            pixels,
            num_patches_list=[len(pixels)],
            questions=[prompt],
            generation_config=generation_config,
        )
    return responses[0].strip() if responses else ""


def output_path(args, record):
    return Path(args.output_dir) / f"{record['video_name']}.json"


def initial_result(args, record):
    answer_model = (
        args.holmes_model_path
        if args.answer_backend == "holmes"
        else args.internvl_model_path
    )
    return {
        "video_name": record["video_name"],
        "dataset": record["dataset"],
        "source_video": record["source_video"],
        "frame_indices": record["mif_frame_selection"]["indices"],
        "narrative": "",
        "answers": [],
        "models": {
            "llama": args.llama_model_path,
            "answer_backend": args.answer_backend,
            "answer_model": answer_model,
        },
    }


def load_result(args, record):
    path = output_path(args, record)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = initial_result(args, record)
    if args.narrative_source_dir:
        source_path = Path(args.narrative_source_dir) / f"{record['video_name']}.json"
        if source_path.exists():
            source = json.loads(source_path.read_text(encoding="utf-8"))
            result["narrative"] = source.get("narrative", "")
    return result


def release(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/vau_hivau/vad_agent_question_conditioned/manifest.jsonl")
    parser.add_argument("--output-dir", default="results/vau_hivau/vad_agent_question_conditioned/results")
    parser.add_argument("--narrative-source-dir")
    parser.add_argument("--video")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--stage", choices=("all", "narrative", "answer"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llama-model-path", default=r"D:\Moshira Research\models\Llama-2-13b-chat-hf")
    parser.add_argument("--llama-dtype", choices=("bf16", "fp16", "fp32"), default="fp16")
    parser.add_argument("--llama-device-map", default="auto")
    parser.add_argument("--llama-max-input-tokens", type=int, default=4096)
    parser.add_argument("--llama-max-new-tokens", type=int, default=220)
    parser.add_argument("--answer-backend", choices=("internvl", "holmes"), default="internvl")
    parser.add_argument("--internvl-model-path", default="OpenGVLab/InternVL2-8B")
    parser.add_argument("--internvl-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--holmes-model-path",
        default="HolmesVAU-2B",
    )
    parser.add_argument("--holmes-max-new-tokens", type=int, default=1024)
    parser.add_argument("--judgement-max-new-tokens", type=int, default=48)
    parser.add_argument("--description-max-new-tokens", type=int, default=120)
    parser.add_argument("--analysis-max-new-tokens", type=int, default=100)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    records = load_jsonl(args.manifest)
    if args.video:
        records = [record for record in records if record["video_name"].lower() == args.video.lower()]
    if args.max_videos:
        records = records[: args.max_videos]
    if not records:
        raise ValueError("No manifest records matched the filters")

    if args.stage in ("all", "narrative"):
        pending = [
            record for record in records
            if args.overwrite or not load_result(args, record).get("narrative")
        ]
        if pending:
            print(f"loading_llama pending_videos={len(pending)}", flush=True)
            model, tokenizer = load_llama(args)
            for record in pending:
                result = initial_result(args, record) if args.overwrite else load_result(args, record)
                result["narrative"] = generate_narrative(model, tokenizer, record, args)
                atomic_write(output_path(args, record), result)
                print(f"narrative_done video={record['video_name']}", flush=True)
            release(model, tokenizer)

    if args.stage in ("all", "answer"):
        pending = []
        for record in records:
            result = load_result(args, record)
            if not result.get("narrative"):
                raise ValueError(f"Missing narrative for {record['video_name']}; run --stage narrative")
            done = {str(item["id"]) for item in result.get("answers", []) if item.get("answer")}
            if args.overwrite or any(str(question["id"]) not in done for question in record["questions"]):
                pending.append(record)
        if pending:
            print(
                f"loading_{args.answer_backend} pending_videos={len(pending)}",
                flush=True,
            )
            if args.answer_backend == "holmes":
                model, tokenizer = load_holmes(args)
            else:
                model, tokenizer = load_internvl(args)
            transform = image_transform()
            for record in pending:
                result = load_result(args, record)
                if args.overwrite:
                    result["answers"] = []
                by_id = {str(item["id"]): item for item in result.get("answers", [])}
                pixels = load_pixels(record, transform)
                for question in record["questions"]:
                    question_id = str(question["id"])
                    if question_id in by_id and by_id[question_id].get("answer") and not args.overwrite:
                        continue
                    answer_record = {
                        "id": question_id,
                        "task": question["task"],
                        "prompt": question["prompt"],
                        "answer": answer_question(
                            model, tokenizer, pixels, result["narrative"], question, args
                        ),
                        "reference": question["reference"],
                    }
                    if args.answer_backend == "internvl":
                        answer_record["output_instruction"] = TASK_OUTPUT_INSTRUCTIONS.get(
                            str(question.get("task") or "").strip().lower(),
                            "Answer the question directly and concisely.",
                        )
                    by_id[question_id] = answer_record
                    result["answers"] = [by_id[key] for key in sorted(by_id, key=int)]
                    atomic_write(output_path(args, record), result)
                    print(
                        f"answer_done video={record['video_name']} question={question_id} "
                        f"task={question['task']}",
                        flush=True,
                    )
                del pixels
            release(model, tokenizer)
    print(f"completed_videos={len(records)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()



