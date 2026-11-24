from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FrameRef:
    index: int
    path: Path

    def to_json(self) -> Dict[str, Any]:
        return {"index": self.index, "path": str(self.path)}


@dataclass
class SegmentPlan:
    segment_id: int
    start: int
    end: int
    context_frames: List[FrameRef]
    selected_frames: List[FrameRef]
    selection_scores: Dict[int, float]

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["context_frames"] = [frame.to_json() for frame in self.context_frames]
        data["selected_frames"] = [frame.to_json() for frame in self.selected_frames]
        data["selection_scores"] = {str(k): v for k, v in self.selection_scores.items()}
        return data


@dataclass
class SegmentResult:
    plan: SegmentPlan
    context_prompt: str
    context: str
    narrative_prompt: str
    narrative_memory: str
    reasoning_prompt: str
    reasoning_response: str
    anomaly_score: Optional[float]
    explanation: str
    timing_seconds: Dict[str, float]

    def to_json(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_json(),
            "context_prompt": self.context_prompt,
            "context": self.context,
            "narrative_prompt": self.narrative_prompt,
            "narrative_memory": self.narrative_memory,
            "reasoning_prompt": self.reasoning_prompt,
            "reasoning_response": self.reasoning_response,
            "anomaly_score": self.anomaly_score,
            "explanation": self.explanation,
            "timing_seconds": self.timing_seconds,
            "model_outputs": {
                "context_agent": self.context,
                "narrative_agent_memory": self.narrative_memory,
                "reasoning_agent": self.reasoning_response,
            },
        }


@dataclass
class VideoResult:
    video_name: str
    num_frames: int
    segments: List[SegmentResult]

    def to_json(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "num_frames": self.num_frames,
            "segments": [segment.to_json() for segment in self.segments],
        }

