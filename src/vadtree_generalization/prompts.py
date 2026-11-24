from __future__ import annotations

from .schemas import SegmentPlan

UCF_EVENTS = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

XD_EVENTS = ["Abuse", "Explosion", "Fighting", "Car Accident", "Shooting", "Riot"]

GENERIC_EVENTS = [
    "physical violence",
    "traffic accident",
    "fire or explosion",
    "weapon use",
    "theft or robbery",
    "vandalism",
    "crowd panic",
    "unusual object or person motion",
]


def anomaly_events(dataset: str) -> list[str]:
    dataset = dataset.lower()
    if dataset == "ucf":
        return UCF_EVENTS
    if dataset == "xd":
        return XD_EVENTS
    return GENERIC_EVENTS


def _event_text(dataset: str) -> str:
    return ", ".join(anomaly_events(dataset))


def _anomaly_questions(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "ucf":
        return """1. Do the current frames show Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, or Vandalism cues?
2. Are people physically restrained, attacked, chased, stealing, hiding items, breaking property, using weapons, or fleeing?
3. Are vehicles damaged, colliding, abruptly stopped, overturned, or involved in unsafe road behavior?
4. Are there visible fire, smoke, blast effects, debris, damage, crowd panic, or sudden abnormal motion?
5. Does the narrative memory make this segment more suspicious than it appears in isolation?"""
    if dataset == "xd":
        return """1. Do the current frames show Abuse, Explosion, Fighting, Car Accident, Shooting, or Riot cues?
2. Are people hitting, grappling, collapsing, aiming weapon-like objects, fleeing, or moving chaotically?
3. Are vehicles damaged, colliding, abruptly stopped, or surrounded by crash-like aftermath?
4. Are there visible fire, smoke, blast effects, debris, large crowd disorder, or police/protest confrontation?
5. Does the narrative memory make this segment more suspicious than it appears in isolation?"""
    return """1. Are people, vehicles, or objects in abnormal positions or being used in abnormal ways?
2. Are there sudden, violent, chaotic, unsafe, or suspicious actions?
3. Is there visible damage, collision, fire, smoke, weapon use, theft, pursuit, or panic?
4. Does this segment change the story in a way that suggests an abnormal event?
5. Can the visual cues be explained as normal activity, camera motion, blur, or sampling ambiguity?"""


def build_context_prompt(dataset: str, plan: SegmentPlan) -> str:
    return f"""You are the model.
** Model Description: **
You are designed to give the overall context of the current video segment using a set of sampled frames from that segment.
Your goal is to describe the segment scene, setting, typical activities, and any recurring objects in a concise, neutral way.
** Prompt Questions: **
Describe the current segment context based on the frames provided.
Focus on scene setting, typical activities, main entities, and the general situation.
** Input: **
[$Data]
Please give your output strictly in the following format:
```
Context Summary: [2-3 sentences describing the overall scene and typical activities.]
Key Entities: [comma-separated list of main people/objects/vehicles seen; use "none" if not visible]
Scene Setting: [one short phrase, e.g., "parking lot", "office hallway", "street outside a gate"]
```
Please ONLY reply according to this format, don't give me any other words.
"""


def build_narrative_prompt(previous_memory: str, current_context: str) -> str:
    return f"""You are the Narrative Agent.

Your task is to combine the previous segments story and the current segment context into one updated story memory.
Write it as a concise temporal story. For example: "First, a person was standing in a parking lot. Then, the person was seen running away. Finally, the person disappeared into the crowd."
Use this only as a style example; do not copy details from it.

Do not infer hidden intent or add details not present in the text.
Keep the answer concise.

Previous story summary:
{previous_memory or "[empty]"}

Current segment context:
{current_context}

Reply only in this exact format:
Story Summary: [1-3 concise sentences summarizing the previous story plus current context]
Key Entities: [comma-separated people/objects/vehicles mentioned; use "none" if not visible]
Scene Setting: [one short phrase]
Open Risk Cues: [brief visible risk or uncertainty cues only; use "none" if no visible risk]
"""


def build_narrative_prompt_coarse(previous_memory: str, current_context: str) -> str:
    return f"""You are the Narrative Agent.

You update a concise story memory for video anomaly detection using text context only.
Merge the previous memory with the current context. Do not replace old useful facts unless the current context contradicts them.
Do not describe your reasoning process. Do not include chain-of-thought. Do not say "I".
Do not add assumptions, hidden intent, or details not present in the context.
Do not include introductions such as "Sure" or "Here is".
Keep the whole answer under 120 words.

Previous memory:
{previous_memory or "[empty]"}

Current context:
{current_context}

Reply only in this exact format:
Story Summary: [1-2 concise sentences describing what has happened so far]
Key Entities: [comma-separated recurring people/objects/vehicles; use "none" if not visible]
Scene Setting: [one short phrase]
Open Risk Cues: [brief visible uncertainty/risk cues only; use "none" if no visible risk]
"""


def build_narrative_prompt_variant(previous_memory: str, current_context: str, variant: str = "current") -> str:
    if variant == "coarse":
        return build_narrative_prompt_coarse(previous_memory, current_context)
    if variant != "current":
        raise ValueError(f"Unknown narrative prompt variant: {variant}")
    return build_narrative_prompt(previous_memory, current_context)


def build_initial_narrative_memory(current_context: str) -> str:
    summary = _extract_context_field(current_context, "Context Summary") or current_context.strip()
    entities = _extract_context_field(current_context, "Key Entities") or "none"
    setting = _extract_context_field(current_context, "Scene Setting") or "unknown"
    return f"""Story Summary: {summary}
Key Entities: {entities}
Scene Setting: {setting}
Open Risk Cues: none"""


def _extract_context_field(text: str, field: str) -> str:
    prefix = f"{field}:"
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def normalize_narrative_memory(text: str, previous_memory: str, current_context: str) -> str:
    summary = _extract_any_field(text, "Story Summary")
    entities = _extract_any_field(text, "Key Entities")
    setting = _extract_any_field(text, "Scene Setting")
    risk = _extract_any_field(text, "Open Risk Cues")
    previous_summary = _extract_any_field(previous_memory, "Story Summary")
    current_summary = _extract_context_field(current_context, "Context Summary")

    if not summary:
        summary = _one_line(" ".join(part for part in (previous_summary, current_summary) if part)) or "none"
    if not entities:
        entities = _extract_context_field(current_context, "Key Entities") or _extract_any_field(previous_memory, "Key Entities") or "none"
    if not setting:
        setting = _extract_context_field(current_context, "Scene Setting") or _extract_any_field(previous_memory, "Scene Setting") or "unknown"
    if not risk:
        risk = _extract_any_field(previous_memory, "Open Risk Cues") or "none"

    return "\n".join(
        [
            f"Story Summary: {_one_line(summary or 'none')}",
            f"Key Entities: {_one_line(entities or 'none')}",
            f"Scene Setting: {_one_line(setting or 'unknown')}",
            f"Open Risk Cues: {_one_line(risk or 'none')}",
        ]
    )


def _extract_any_field(text: str, field: str) -> str:
    prefix = f"{field}:"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        value = stripped.split(":", 1)[1].strip()
        extra = []
        for next_line in lines[idx + 1 :]:
            next_stripped = next_line.strip()
            if _is_memory_field(next_stripped):
                break
            if next_stripped:
                extra.append(next_stripped)
        return " ".join([value, *extra]).strip()
    return ""


def _is_memory_field(line: str) -> bool:
    return any(
        line.startswith(field)
        for field in ("Story Summary:", "Key Entities:", "Scene Setting:", "Open Risk Cues:")
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def build_reasoning_prompt(dataset: str, previous_memory: str, plan: SegmentPlan, score_mode: str = "anomaly") -> str:
    if score_mode == "binary":
        return build_binary_reasoning_prompt(dataset, previous_memory)
    return f"""You are designed to do binary classification for video anomaly detection.
The input is a set of frames from the current video segment and a short story describing what happened before this segment.
The previous story is provided for you to preserve temporal dependency across the video.
First decide the visual reason based on the current frames and previous story, then assign an anomaly score that is consistent with that reason.
If the reason contains suspicious, unsafe, violent, theft-like, panic, damage, weapon, fire, collision, or fleeing cues, the anomaly score should not be very low unless the evidence is clearly ambiguous or normal.

Use anomaly_score where 0 means normal and 1 means anomalous.
A normal reason must have a low score near 0.
A clearly anomalous reason must have a high score near 1.

Do not invert the scale. A normal reason must have a low score. A clearly anomalous reason must have a high score.

Anomaly-aware questions:
{_anomaly_questions(dataset)}

Dataset abnormal event priors: {_event_text(dataset)}.

Previous segments story:
{previous_memory or "[empty]"}

Return strict JSON:
{{
  "reason": "brief evidence-based reason from the current frames and previous story",
  "anomaly_score": number between 0 and 1
}}
"""


def build_binary_reasoning_prompt(dataset: str, previous_memory: str) -> str:
    return f"""You are designed to do binary classification for video anomaly detection.
The input is a set of frames from the current video segment and a short story describing what happened before this segment.
The previous story is provided for you to preserve temporal dependency across the video.
First decide the visual reason based on the current frames and previous story, then assign a binary anomaly label that is consistent with that reason.
If the reason contains suspicious, unsafe, violent, theft-like, panic, damage, weapon, fire, collision, or fleeing cues, the anomaly label should not be 0 unless the evidence is clearly ambiguous or normal.

Use anomaly_score where 0 means normal and 1 means anomalous.
A normal reason must have label 0.
A clearly anomalous reason must have label 1.


Anomaly-aware questions:
{_anomaly_questions(dataset)}

Dataset abnormal event priors: {_event_text(dataset)}.

Previous segments story:
{previous_memory or "[empty]"}

Return strict JSON:
{{
  "reason": "brief evidence-based reason from the current frames and previous story",
  "anomaly_score": 0 or 1
}}
"""

