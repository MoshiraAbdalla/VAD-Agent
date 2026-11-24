from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol


class VLMClient(Protocol):
    """Visual-language model boundary used by context and reasoning agents."""

    def generate(self, prompt: str, image_paths: List[Path]) -> str:
        ...


class LLMClient(Protocol):
    """Text-only LLM boundary used by the narrative memory agent."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class TransformersLLMClient:
    """Local Hugging Face causal-LM client for the narrative agent."""

    checkpoint: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    device_map: str = "auto"
    torch_dtype: str = "auto"
    strip_thinking: bool = True
    force_no_think: bool = False
    system_prompt: str | None = None
    chat_format: str = "auto"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _resolve_torch_dtype(torch, self.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        if self.chat_format == "llama2":
            system = self.system_prompt or "You are a helpful assistant."
            text = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt.strip()} [/INST]"
        elif hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        if self.force_no_think:
            text = text + "\n</think>\n\n"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return _strip_thinking(text) if self.strip_thinking else text


@dataclass
class LlavaVideoClient:
    """Local LLaVA-Video client for context and reasoning agents."""

    pretrained: str
    llava_repo: Path | None = None
    model_base: str | None = None
    model_name: str = "llava_qwen"
    device: str = "cuda"
    device_map: str = "auto"
    conv_template: str = "qwen_1_5"
    max_new_tokens: int = 512
    torch_dtype: str = "auto"
    attn_implementation: str = "sdpa"
    load_8bit: bool = False
    load_4bit: bool = False

    def __post_init__(self) -> None:
        repo_path = self.llava_repo or Path(__file__).resolve().parents[1] / "LLaVA-NeXT"
        if repo_path.exists():
            sys.path.insert(0, str(repo_path))

        import torch
        from llava.model.builder import load_pretrained_model

        dtype = _resolve_llava_dtype(torch, self.torch_dtype)
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device_map,
            "load_8bit": self.load_8bit,
            "load_4bit": self.load_4bit,
        }
        load_kwargs["attn_implementation"] = self.attn_implementation

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.pretrained,
            self.model_base,
            self.model_name,
            **load_kwargs,
        )
        self.model.eval()

    def generate(self, prompt: str, image_paths: List[Path]) -> str:
        import copy

        import numpy as np
        import torch
        from PIL import Image

        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        frames = []
        for path in image_paths:
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("RGB")))
        if not frames:
            raise ValueError("LLaVA received no frames.")

        video_np = np.stack(frames, axis=0)
        pixel_values = self.image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"]
        if self.device.startswith("cuda") and torch.cuda.is_available():
            pixel_values = pixel_values.half().to(self.device)
        else:
            pixel_values = pixel_values.float()

        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_question,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)

        model_device = next(self.model.parameters()).device
        target_device = self.device if self.device.startswith("cuda") and torch.cuda.is_available() else model_device
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids.to(target_device),
                images=[pixel_values.to(target_device)],
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=self.max_new_tokens,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


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


def _resolve_llava_dtype(torch_module, dtype: str) -> str:
    if dtype == "auto":
        return "float16" if torch_module.cuda.is_available() else "float32"
    if dtype in {"float16", "fp16"}:
        return "float16"
    if dtype in {"bfloat16", "bf16"}:
        return "bfloat16"
    if dtype in {"float32", "fp32"}:
        return "float32"
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text.strip()


def parse_reasoning_json(text: str) -> tuple[float | None, str]:
    """Parse the VLM reasoning JSON while tolerating markdown fences or extra text."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        score = _extract_reasoning_score(payload)
        explanation = _extract_reasoning_explanation(payload) or text.strip()
        return score, explanation

    score = data.get("anomaly_score")
    explanation = data.get("reason") or data.get("explanation") or json.dumps(data, ensure_ascii=False)
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    return score, str(explanation)


def parse_binary_reasoning(text: str) -> tuple[float | None, str]:
    """Parse binary reasoning where 0=normal and 1=abnormal."""
    stripped = str(text or "").strip()
    if stripped in {"0", "1"}:
        score = float(stripped)
        return score, "abnormal" if score == 1.0 else "normal"

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    payload = match.group(0) if match else stripped
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        raw_score = data.get("anomaly_score", data.get("anomaly_label", data.get("label")))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = None
        if score in {0.0, 1.0}:
            explanation = data.get("reason") or data.get("explanation")
            return score, str(explanation or ("abnormal" if score == 1.0 else "normal"))

    score = _extract_reasoning_score(payload)
    if score in {0.0, 1.0}:
        explanation = _extract_reasoning_explanation(payload)
        return score, explanation or ("abnormal" if score == 1.0 else "normal")

    match = re.search(r"\b([01])\b", stripped)
    if not match:
        return None, stripped
    score = float(match.group(1))
    return score, "abnormal" if score == 1.0 else "normal"


def _extract_reasoning_score(text: str) -> float | None:
    match = re.search(
        r'"?anomaly_score"?\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_reasoning_explanation(text: str) -> str:
    # Handles malformed JSON where the reason contains unescaped quotes, as long as
    # the reason field still appears near anomaly_score.
    match = re.search(
        r'"(?:reason|explanation)"\s*:\s*"(.*?)"\s*,\s*"?anomaly_score"?\s*:',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r'"?anomaly_score"?\s*:\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"(?:reason|explanation)"\s*:\s*"(.*?)"',
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    if not match:
        match = re.search(
            r'"(?:reason|explanation)"\s*:\s*"(.*?)"\s*(?:\n|\r|\})',
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    if not match:
        return ""
    explanation = match.group(1)
    explanation = explanation.replace('\\"', '"').replace("\\n", " ")
    return re.sub(r"\s+", " ", explanation).strip()

