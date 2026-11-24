from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .sampling import select_feature_local_minima


@dataclass
class ImageBindFrameSelector:
    imagebind_repo: Path | None = None
    checkpoint: Path | None = None
    device: str = "cuda"
    chunk_size: int = 128
    window: int = -1
    alpha: float = 2.5
    feature_dir: Path | None = None
    dataset_name: str = "generic"

    def __post_init__(self) -> None:
        repo_path = self.imagebind_repo or Path(__file__).resolve().parents[1] / "ImageBind"
        if repo_path.exists():
            sys.path.insert(0, str(repo_path))

        import torch
        from imagebind.models import imagebind_model

        self.torch = torch
        self.runtime_device = self.device if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=self.checkpoint is None)
        if self.checkpoint is not None:
            state = torch.load(str(self.checkpoint), map_location="cpu")
            self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.runtime_device)

    def select(
        self,
        frame_paths: Sequence[Path],
        num_samples: int,
        cache_info: Dict[str, Any] | None = None,
    ) -> Tuple[List[int], Dict[int, float]]:
        features = self._load_or_extract_features(frame_paths, cache_info)
        selected, scores = select_feature_local_minima(
            features,
            num_samples=num_samples,
            window=self.window,
            alpha=self.alpha,
        )
        self._write_feature_metadata(frame_paths, cache_info, features, selected, scores)
        return selected, scores

    def extract_features(self, frame_paths: Sequence[Path]) -> np.ndarray:
        if len(frame_paths) == 0:
            return np.empty((0, 0), dtype=np.float32)

        from imagebind import data
        from imagebind.models.imagebind_model import ModalityType

        all_features = []
        with self.torch.no_grad():
            for start in range(0, len(frame_paths), self.chunk_size):
                chunk_paths = [str(path) for path in frame_paths[start : start + self.chunk_size]]
                inputs = data.load_and_transform_vision_data(chunk_paths, self.runtime_device)
                outputs = self.model({ModalityType.VISION: inputs})[ModalityType.VISION]
                all_features.append(outputs.detach().float().cpu())
                del inputs, outputs
                if self.runtime_device.startswith("cuda"):
                    self.torch.cuda.empty_cache()

        return self.torch.cat(all_features, dim=0).numpy().astype(np.float32, copy=False)

    def _load_or_extract_features(
        self,
        frame_paths: Sequence[Path],
        cache_info: Dict[str, Any] | None,
    ) -> np.ndarray:
        feature_path, meta_path = self._feature_paths(frame_paths, cache_info)
        if feature_path is not None and meta_path is not None and feature_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("frame_paths") == [str(path) for path in frame_paths]:
                    with np.load(feature_path) as data:
                        features = data["features"].astype(np.float32, copy=False)
                    if features.shape[0] == len(frame_paths):
                        return features
            except Exception as exc:
                print(f"[ImageBind cache] ignoring unreadable cache {feature_path}: {exc}", flush=True)

        features = self.extract_features(frame_paths)
        if feature_path is not None:
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(feature_path, features=features)
        return features

    def _write_feature_metadata(
        self,
        frame_paths: Sequence[Path],
        cache_info: Dict[str, Any] | None,
        features: np.ndarray,
        selected: Sequence[int],
        scores: Dict[int, float],
    ) -> None:
        feature_path, meta_path = self._feature_paths(frame_paths, cache_info)
        if meta_path is None:
            return

        global_indices = _global_frame_indices(frame_paths, cache_info)
        selected_global = [
            global_indices[idx] if idx < len(global_indices) else idx
            for idx in selected
        ]
        payload = {
            "dataset": self.dataset_name,
            "video_name": _video_name(frame_paths),
            "segment_id": None if cache_info is None else cache_info.get("segment_id"),
            "start": None if cache_info is None else cache_info.get("start"),
            "end": None if cache_info is None else cache_info.get("end"),
            "feature_file": str(feature_path) if feature_path is not None else None,
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "frame_paths": [str(path) for path in frame_paths],
            "global_frame_indices": global_indices,
            "selected_local_indices": [int(idx) for idx in selected],
            "selected_global_indices": [int(idx) for idx in selected_global],
            "selection_scores_local": {str(k): float(v) for k, v in scores.items()},
            "window": self.window,
            "alpha": self.alpha,
            "chunk_size": self.chunk_size,
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _feature_paths(
        self,
        frame_paths: Sequence[Path],
        cache_info: Dict[str, Any] | None,
    ) -> Tuple[Path | None, Path | None]:
        if self.feature_dir is None:
            return None, None
        video_name = _video_name(frame_paths)
        segment_id = 0 if cache_info is None else int(cache_info.get("segment_id", 0))
        start = 0 if cache_info is None else int(cache_info.get("start", 0))
        end = max(0, len(frame_paths) - 1) if cache_info is None else int(cache_info.get("end", max(0, len(frame_paths) - 1)))
        stem = f"segment_{segment_id:06d}_{start}_{end}"
        out_dir = self.feature_dir / self.dataset_name / video_name
        return out_dir / f"{stem}.npz", out_dir / f"{stem}.json"


def _video_name(frame_paths: Sequence[Path]) -> str:
    if len(frame_paths) == 0:
        return "unknown_video"
    return frame_paths[0].parent.name


def _global_frame_indices(frame_paths: Sequence[Path], cache_info: Dict[str, Any] | None) -> List[int]:
    if cache_info is not None and "global_frame_indices" in cache_info:
        return [int(idx) for idx in cache_info["global_frame_indices"]]
    return list(range(len(frame_paths)))

