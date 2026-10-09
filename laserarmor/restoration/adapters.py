from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .dominant_remove import StripeResDominantRemove
from .residual_remove import build_residual_model


def mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None:
        return None
    rows, columns = np.where(np.asarray(mask) > 0)
    if not len(rows):
        return None
    return int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1


def extract_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict", "model", "net_g"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and candidate:
                checkpoint = candidate
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint does not contain a state dictionary")
    state = {str(key): value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)}
    if not state:
        raise RuntimeError("Checkpoint state dictionary is empty")
    for prefix in ("module.", "model.", "net."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def resolve_asset(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"Required asset was not found: {path}")
    return path


class TorchImageRestorer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        name: str,
        rgb: bool,
        pad_multiple: int,
        amp: bool,
        square: bool = False,
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.name = name
        self.rgb = bool(rgb)
        self.pad_multiple = max(1, int(pad_multiple))
        self.amp = bool(amp and device.type == "cuda")
        self.square = bool(square)

    def _pad(self, tensor: torch.Tensor):
        height, width = tensor.shape[-2:]
        target_height = ((height + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        target_width = ((width + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        if self.square:
            side = max(target_height, target_width)
            target_height = side
            target_width = side
        bottom = target_height - height
        right = target_width - width
        if bottom or right:
            can_reflect = height > 1 and width > 1 and bottom < height and right < width
            mode = "reflect" if can_reflect else "replicate"
            tensor = F.pad(tensor, (0, right, 0, bottom), mode=mode)
        return tensor, (height, width)

    def restore(self, image_bgr: np.ndarray, mask: np.ndarray | None = None, spans=None):
        del spans
        bbox = mask_bbox(mask)
        if bbox is None:
            x0, y0, x1, y1 = 0, 0, image_bgr.shape[1], image_bgr.shape[0]
        else:
            x0, y0, x1, y1 = bbox
        roi = image_bgr[y0:y1, x0:x1]
        model_input = roi[..., ::-1].copy() if self.rgb else roi.copy()
        tensor = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(self.device)
        tensor, original_shape = self._pad(tensor)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            prediction = self.model(tensor)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        prediction = prediction[..., : original_shape[0], : original_shape[1]]
        restored = prediction.float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
        restored = np.clip(np.round(restored * 255.0), 0, 255).astype(np.uint8)
        if self.rgb:
            restored = restored[..., ::-1].copy()
        output = image_bgr.copy()
        if mask is None or bbox is None:
            output[y0:y1, x0:x1] = restored
        else:
            local_mask = np.asarray(mask[y0:y1, x0:x1]) > 0
            local_output = output[y0:y1, x0:x1]
            local_output[local_mask] = restored[local_mask]
            output[y0:y1, x0:x1] = local_output
        return output, {"backend": self.name, "bbox": bbox}


class ADOMRestorer:
    def restore(self, image_bgr: np.ndarray, mask: np.ndarray | None = None, spans=None):
        from .backends.ADOM.model import ADOM

        del spans
        bbox = mask_bbox(mask)
        if bbox is None:
            x0, y0, x1, y1 = 0, 0, image_bgr.shape[1], image_bgr.shape[0]
        else:
            x0, y0, x1, y1 = bbox
        roi = image_bgr[y0:y1, x0:x1]
        result = ADOM(roi).process()
        restored = np.asarray(result.get("destriped", roi), dtype=np.uint8)
        if restored.ndim == 2:
            restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
        if restored.shape[:2] != roi.shape[:2]:
            restored = cv2.resize(restored, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        output = image_bgr.copy()
        output[y0:y1, x0:x1] = restored
        if mask is not None:
            selected = np.asarray(mask) > 0
            blended = image_bgr.copy()
            blended[selected] = output[selected]
            output = blended
        return output, {"backend": "ADOM", "bbox": bbox}


class ShiftLUTRestorer:
    def __init__(self, lut_path: Path, device: torch.device) -> None:
        from .backends.ShiftLUT.model import LUT, inference

        self.model = LUT(lut_path).eval().to(device)
        self.inference = inference
        self.device = device
        self.lut_path = lut_path

    def restore(self, image_bgr: np.ndarray, mask: np.ndarray | None = None, spans=None):
        del spans
        bbox = mask_bbox(mask)
        if bbox is None:
            x0, y0, x1, y1 = 0, 0, image_bgr.shape[1], image_bgr.shape[0]
        else:
            x0, y0, x1, y1 = bbox
        roi = image_bgr[y0:y1, x0:x1]
        tensor = torch.from_numpy(roi.copy()).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            restored_tensor = self.inference(self.model, tensor - 128.0) + tensor
        restored = restored_tensor.clamp(0, 255)[0].permute(1, 2, 0).round().byte().cpu().numpy()
        output = image_bgr.copy()
        if mask is None or bbox is None:
            output[y0:y1, x0:x1] = restored
        else:
            selected = np.asarray(mask[y0:y1, x0:x1]) > 0
            local = output[y0:y1, x0:x1]
            local[selected] = restored[selected]
            output[y0:y1, x0:x1] = local
        return output, {"backend": "ShiftLUT", "bbox": bbox, "lut_path": str(self.lut_path)}


class TSFormerRestorer:
    def __init__(self, artifact: Path, device: torch.device, config: dict[str, Any]) -> None:
        from .backends.TSFormer.inference import TSFormerImageRestorer, load_restoration_backend

        backend, _ = load_restoration_backend(
            artifact_path=artifact,
            device=device,
            runtime=str(config.get("runtime", "auto")),
            use_half=bool(config.get("amp", False)),
            batch_size=1,
            config_override=dict(config.get("model", {})),
        )
        self.restorer = TSFormerImageRestorer(backend)

    def restore(self, image_bgr: np.ndarray, mask: np.ndarray | None = None, spans=None):
        del spans
        restored, info = self.restorer.restore_image_on_mask(image_bgr, mask=mask, blend_mask=mask)
        info.pop("model_input_bgr", None)
        info["backend"] = "TSFormer"
        return restored, info


class StripeResPipeline:
    def __init__(self, dominant: StripeResDominantRemove, residual: TorchImageRestorer) -> None:
        self.dominant = dominant
        self.residual = residual

    def restore(self, image_bgr: np.ndarray, mask: np.ndarray | None = None, spans=None):
        coarse, coarse_info = self.dominant.restore(image_bgr, mask=mask, spans=spans)
        restored, residual_info = self.residual.restore(coarse, mask=mask, spans=spans)
        return restored, {"backend": "StripeRes", "coarse": coarse_info, "residual": residual_info}


def load_model_state(model: torch.nn.Module, checkpoint_path: Path, strict: bool) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(extract_state(checkpoint), strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def build_restorer(
    name: str,
    config: dict[str, Any],
    device: torch.device,
    project_root: Path,
):
    key = str(name).strip().lower()
    amp = bool(config.get("amp", False))
    if key == "adom":
        return ADOMRestorer()
    if key == "striperes-dominantremove":
        return StripeResDominantRemove(**dict(config.get("parameters", {})))
    if key in {"striperes-residualremove", "striperes"}:
        checkpoint_path = resolve_asset(project_root, config["checkpoint"])
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        training = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
        variant = str(training.get("lightweight_variant", "row_lut")) if isinstance(training, dict) else "row_lut"
        if variant != "row_lut":
            raise RuntimeError(f"StripeRes-ResidualRemove requires a row_lut checkpoint, got {variant}")
        model = build_residual_model(checkpoint)
        model.load_state_dict(extract_state(checkpoint), strict=True)
        residual = TorchImageRestorer(model, device, "StripeRes-ResidualRemove", True, 1, amp)
        if key == "striperes-residualremove":
            return residual
        dominant = StripeResDominantRemove(**dict(config.get("dominant_parameters", {})))
        return StripeResPipeline(dominant, residual)
    if key == "tsformer":
        return TSFormerRestorer(resolve_asset(project_root, config["checkpoint"]), device, config)
    if key == "shiftlut":
        return ShiftLUTRestorer(resolve_asset(project_root, config["lut_path"]), device)
    checkpoint_path = resolve_asset(project_root, config["checkpoint"])
    if key == "restormer":
        from .backends.Restormer.model import Restormer

        parameters = dict(config.get("parameters", {}))
        model = Restormer(**parameters)
        load_model_state(model, checkpoint_path, True)
        return TorchImageRestorer(model, device, "Restormer", True, 8, amp)
    if key == "uformer":
        from .backends.Uformer.model import Uformer

        parameters = dict(config.get("parameters", {}))
        model = Uformer(**parameters)
        load_model_state(model, checkpoint_path, True)
        return TorchImageRestorer(model, device, "Uformer", True, 128, amp, square=True)
    if key == "nafnet":
        from .backends.NAFNet.model import NAFNet

        parameters = dict(config.get("parameters", {}))
        model = NAFNet(**parameters)
        load_model_state(model, checkpoint_path, False)
        return TorchImageRestorer(model, device, "NAFNet", False, 16, amp)
    if key == "anyir":
        from .backends.AnyIR.model import AnyIR

        parameters = dict(config.get("parameters", {}))
        model = AnyIR(**parameters)
        load_model_state(model, checkpoint_path, True)
        return TorchImageRestorer(model, device, "AnyIR", True, 8, amp)
    if key == "mpmf-net":
        from .backends.MPMFNet.model import MPMFNet

        model = MPMFNet()
        load_model_state(model, checkpoint_path, True)
        return TorchImageRestorer(model, device, "MPMF-Net", True, 4, amp)
    if key == "dcmgformer":
        from .backends.DCMGFormer.model import DCMGFormer

        model = DCMGFormer()
        load_model_state(model, checkpoint_path, True)
        return TorchImageRestorer(model, device, "DCMGFormer", True, 16, amp)
    if key == "jarvisir":
        from .backends.JarvisIR.model import SCUNet

        parameters = dict(config.get("parameters", {}))
        model = SCUNet(**parameters)
        load_model_state(model, checkpoint_path, True)
        return TorchImageRestorer(model, device, "JarvisIR", True, 64, amp)
    if key == "gendeg":
        from .backends.GenDeg.model import NAFNet as GenDegNAFNet

        parameters = dict(config.get("parameters", {}))
        model = GenDegNAFNet(**parameters)
        load_model_state(model, checkpoint_path, True)
        return TorchImageRestorer(model, device, "GenDeg", True, 16, amp)
    raise RuntimeError(f"Unsupported restoration method: {name}")
