from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .model import build_tsformer_from_config


@dataclass
class LoadedTorchModel:
    model: torch.nn.Module
    checkpoint: Dict[str, object]


def _torch_load_optimized(path: str | Path, map_location: str | torch.device = "cpu"):
    load_kwargs = {"map_location": map_location, "weights_only": True, "mmap": True}
    try:
        return torch.load(str(path), **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap", None)
        try:
            return torch.load(str(path), **load_kwargs)
        except TypeError:
            return torch.load(str(path), map_location=map_location)
    except RuntimeError:
        load_kwargs.pop("mmap", None)
        return torch.load(str(path), **load_kwargs)


def _mask_to_bbox(mask: np.ndarray | None) -> Tuple[int, int, int, int] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _blend_restored_pixels(
    base_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    blend_mask: np.ndarray | None,
) -> np.ndarray:
    if blend_mask is None:
        return restored_bgr
    blend_mask = np.where(np.asarray(blend_mask) > 0, 255, 0).astype(np.uint8)
    if not np.any(blend_mask > 0):
        return restored_bgr
    bbox = _mask_to_bbox(blend_mask)
    if bbox is None:
        return restored_bgr

    x0, y0, x1, y1 = bbox
    output = base_bgr.copy()
    keep = blend_mask[y0:y1, x0:x1] > 0
    output_roi = output[y0:y1, x0:x1]
    restored_roi = restored_bgr[y0:y1, x0:x1]
    output_roi[keep] = restored_roi[keep]
    output[y0:y1, x0:x1] = output_roi
    return output


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    use_half: bool = False,
    config_override: Dict[str, object] | None = None,
) -> LoadedTorchModel:
    checkpoint = _torch_load_optimized(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_config = checkpoint.get("model_config", {})
    else:
        state_dict = checkpoint
        model_config = {}

    effective_config = dict(model_config)
    if config_override:
        for key, value in config_override.items():
            if value is not None:
                effective_config[key] = value

    model = build_tsformer_from_config(effective_config)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if use_half and device.type == "cuda":
        model.half()
    metadata = {
        "source_path": str(Path(checkpoint_path).resolve()),
        "model_config": effective_config,
        "artifact_type": str(checkpoint.get("artifact_type", "checkpoint" if "state_dict" in checkpoint else "state_dict")),
    }
    del checkpoint
    return LoadedTorchModel(model=model, checkpoint=metadata)


class TorchBackend:
    def __init__(self, model: torch.nn.Module, device: torch.device, use_half: bool = False, batch_size: int = 1) -> None:
        self.model = model
        self.device = device
        self.use_half = bool(use_half and device.type == "cuda")
        self.batch_size = max(1, int(batch_size))
        self.use_channels_last = device.type == "cuda"
        self.non_blocking = device.type == "cuda"

    def predict(self, images: np.ndarray) -> np.ndarray:
        outputs: List[np.ndarray] = []
        dtype = torch.float16 if self.use_half else torch.float32
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.from_numpy(images[start:start + self.batch_size]).permute(0, 3, 1, 2)
                if self.use_channels_last:
                    batch = batch.contiguous(memory_format=torch.channels_last).pin_memory()
                batch = batch.to(device=self.device, dtype=dtype, non_blocking=self.non_blocking)
                pred = self.model(batch).float().cpu().permute(0, 2, 3, 1).numpy()
                outputs.append(pred)
        return np.concatenate(outputs, axis=0)


class OnnxBackend:
    def __init__(self, model_path: str | Path, device: torch.device, batch_size: int = 1) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to run ONNX model artifacts.") from exc

        available = ort.get_available_providers()
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.batch_size = max(1, int(batch_size))

    def predict(self, images: np.ndarray) -> np.ndarray:
        outputs: List[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = np.transpose(images[start:start + self.batch_size], (0, 3, 1, 2)).astype(np.float32)
            pred = self.session.run([self.output_name], {self.input_name: batch})[0]
            outputs.append(np.transpose(pred, (0, 2, 3, 1)))
        return np.concatenate(outputs, axis=0)


def load_restoration_backend(
    artifact_path: str | Path,
    device: torch.device,
    runtime: str = "auto",
    use_half: bool = False,
    batch_size: int = 1,
    config_override: Dict[str, object] | None = None,
):
    artifact_path = Path(artifact_path)
    runtime = runtime.lower()
    if runtime == "auto":
        runtime = "onnx" if artifact_path.suffix.lower() == ".onnx" else "torch"

    if runtime == "torch":
        loaded = load_checkpoint_model(
            artifact_path,
            device=device,
            use_half=use_half,
            config_override=config_override,
        )
        return TorchBackend(loaded.model, device=device, use_half=use_half, batch_size=batch_size), loaded.checkpoint
    if runtime == "onnx":
        return OnnxBackend(artifact_path, device=device, batch_size=batch_size), {}
    raise RuntimeError(f"Unsupported runtime: {runtime}")


class TSFormerImageRestorer:
    def __init__(self, backend) -> None:
        self.backend = backend

    def restore_image(self, degraded_bgr: np.ndarray) -> np.ndarray:
        pred = self.backend.predict(np.expand_dims(degraded_bgr.astype(np.float32) / 255.0, axis=0))[0]
        pred = np.clip(np.round(pred * 255.0), 0, 255).astype(np.uint8)
        return pred

    def restore_image_on_mask(
        self,
        degraded_bgr: np.ndarray,
        mask: np.ndarray | None = None,
        blend_mask: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        bbox = _mask_to_bbox(mask)
        if bbox is None:
            restored_bgr = self.restore_image(degraded_bgr)
            return _blend_restored_pixels(degraded_bgr, restored_bgr, blend_mask), {
                "infer_mode": "full_image",
                "bbox": None,
                "model_input_bgr": degraded_bgr.copy(),
            }

        x0, y0, x1, y1 = bbox
        roi_input = degraded_bgr[y0:y1, x0:x1]
        roi_output = self.restore_image(roi_input)
        output = degraded_bgr.copy()
        output[y0:y1, x0:x1] = roi_output
        output = _blend_restored_pixels(degraded_bgr, output, blend_mask if blend_mask is not None else mask)
        return output, {
            "infer_mode": "roi_crop",
            "bbox": bbox,
            "model_input_bgr": roi_input.copy(),
        }

    def restore_with_reference(
        self,
        abnormal_bgr: np.ndarray,
        reference_bgr: np.ndarray | None = None,
        stripe_spans: List[Tuple[int, int]] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        del reference_bgr, stripe_spans
        return self.restore_image(abnormal_bgr), {
            "model_input_bgr": abnormal_bgr.copy(),
            "fallback": None,
        }

    def step(
        self,
        frame_bgr: np.ndarray,
        attack_flag: int = 1,
        stripe_spans: List[Tuple[int, int]] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        del stripe_spans
        info: Dict[str, object] = {
            "model_input_bgr": frame_bgr.copy(),
            "fallback": None,
        }
        if not attack_flag:
            return frame_bgr.copy(), info
        return self.restore_image(frame_bgr), info
