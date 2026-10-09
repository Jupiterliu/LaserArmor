from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def LUTclip(x):
    return x.clamp(-128, 127)


def Query(values: torch.Tensor, x: torch.Tensor, sample_d: int):
    if sample_d == 1:
        return values[x]

    a1 = x // sample_d
    a2 = a1 + 1
    weight = (x % sample_d).to(torch.float32) / float(sample_d)

    if values.dim() > 1:
        weight = weight[..., None]

    return values[a1] * (1 - weight) + values[a2] * weight


class dwLUT(nn.Module):
    def __init__(self, path: str | Path):
        super().__init__()
        root = Path(path)
        arrays = np.load(str(root) + ".npz")
        with open(str(root) + ".json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.arr = nn.ParameterDict()
        self.steps = {}
        self.lookup_count = int(self.meta["lookup_count"])
        self.in_channels = int(self.meta["IN"])
        self.out_channels = int(self.meta["OUT"])

        for in_idx in range(self.in_channels):
            for out_idx in range(self.out_channels):
                key = f"i{in_idx}o{out_idx}"
                self.arr[key] = nn.Parameter(torch.from_numpy(arrays[key]).to(torch.int32), requires_grad=False)

        for info in self.meta["LUTs"]:
            self.steps[f'i{info["in"]}o{info["out"]}'] = int(info["step"])

    def forward(self, x, debug=False):
        n, c, h, w = x.shape
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        unfolded = F.unfold(x.to(torch.float32), kernel_size=(3, 3), padding=0).to(torch.int32)
        unfolded = unfolded.view(n, c, 9, -1)

        if c not in (1, self.out_channels):
            raise RuntimeError(f"Unsupported dwLUT input channel count: {c}")

        outputs = []
        for out_idx in range(self.out_channels):
            src_channel = 0 if c == 1 else out_idx
            value = sum(
                Query(self.arr[f"i{kernel_idx}o{out_idx}"], unfolded[:, src_channel, kernel_idx, :], self.steps[f"i{kernel_idx}o{out_idx}"])
                for kernel_idx in range(self.in_channels)
            )
            value = torch.round(value / float(self.in_channels))
            outputs.append(value)

        output = torch.stack(outputs, 1)
        return output.reshape(n, self.out_channels, h, w)


class pwLUT(nn.Module):
    def __init__(self, path: str | Path):
        super().__init__()
        root = Path(path)
        arrays = np.load(str(root) + ".npz")
        with open(str(root) + ".json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.arr = nn.ParameterDict()
        self.steps = {}
        self.in_channels = int(self.meta["IN"])
        self.out_channels = int(self.meta["OUT"])

        for in_idx in range(self.in_channels):
            for out_idx in range(self.out_channels):
                key = f"i{in_idx}o{out_idx}"
                self.arr[key] = nn.Parameter(torch.from_numpy(arrays[key]).to(torch.int32), requires_grad=False)

        for info in self.meta["LUTs"]:
            self.steps[f'i{info["in"]}o{info["out"]}'] = int(info["step"])

    def forward(self, x, debug=False):
        x = x.to(torch.int32)
        n, c, h, w = x.shape
        if c != self.in_channels:
            raise RuntimeError(f"Unexpected pwLUT input channels: {c}, expected {self.in_channels}")

        output = torch.zeros(n, self.out_channels, h, w, device=x.device, dtype=torch.float32)
        for out_idx in range(self.out_channels):
            output[:, out_idx, ...] = torch.round(
                sum(
                    Query(self.arr[f"i{in_idx}o{out_idx}"], x[:, in_idx, :, :], self.steps[f"i{in_idx}o{out_idx}"])
                    for in_idx in range(self.in_channels)
                )
                / float(self.in_channels)
            )
        return output


class LUT(nn.Module):
    def __init__(self, lut_path: str | Path):
        super().__init__()
        self.lut_path = Path(lut_path)
        config_path = self.lut_path / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"LUT config was not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        self.stack = int(config["stack"])
        self.scale = int(config["scale"])
        self.cnum = int(config["cnum"])
        self.msb_base = int(config["msb_base"])
        self.lsb_base = int(config["lsb_base"])
        self.msb_min = -(2 ** (self.msb_base - 1))
        self.msb_max = 2 ** (self.msb_base - 1) - 1
        self.msb_bias = -self.msb_min
        self.lsb_max = 2 ** self.lsb_base - 1

        self.offset_constant = torch.from_numpy(np.load(self.lut_path / "offset.npy")).to(torch.int32)
        self.msb = nn.ModuleList()
        self.lsb = nn.ModuleList()

        for idx in range(self.stack + 1):
            self.msb.append(dwLUT(self.lut_path / f"DW{idx}_MSB"))
            self.msb.append(pwLUT(self.lut_path / f"PW{idx}_MSB"))

        self.lsb.append(dwLUT(self.lut_path / "DW0_LSB"))
        self.msb.append(pwLUT(self.lut_path / "UP_MSB"))
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def low_high(self, image: torch.Tensor):
        image = torch.round(image)
        xl = torch.remainder(image, 2 ** self.lsb_base)
        xh = torch.div(image, 2 ** self.lsb_base, rounding_mode="floor")
        return xl.to(torch.int32), xh.to(torch.int32)

    def shift(self, x: torch.Tensor, constant_offset: torch.Tensor):
        n, c, h, w = x.shape
        shifted = x.clone()

        for channel in range(min(c, constant_offset.shape[-1])):
            dy = int(constant_offset[0, channel].item())
            dx = int(constant_offset[1, channel].item())
            x_pad = F.pad(x[:, channel], pad=(abs(dx), abs(dx), abs(dy), abs(dy)), mode="replicate")
            shifted[:, channel] = x_pad[:, abs(dy) + dy : abs(dy) + dy + h, abs(dx) + dx : abs(dx) + dx + w]

        return shifted

    def forward(self, x, debug=False):
        x_l, x_h = self.low_high(x)
        shift_idx = 0

        for idx in range(len(self.msb) - 1):
            if isinstance(self.msb[idx], pwLUT):
                if shift_idx < self.offset_constant.shape[0]:
                    x_h = self.shift(x_h, self.offset_constant[shift_idx].to(x_h.device))
                shift_idx += 1

            x_h = self.msb[idx](x_h + self.msb_bias, False) + x_h

            if idx == 0:
                x_l = self.lsb[0](x_l, False) + x_l
                x_l.clamp_(0, self.lsb_max)
                x_h += x_l

            x_h.clamp_(self.msb_min, self.msb_max)

        output = self.msb[-1](x_h + self.msb_bias, False)
        output = self.pixel_shuffle(output)
        return output.to(torch.int16)


def inference(module: LUT, x: torch.Tensor):
    n, c, h, w = x.shape
    x = x.reshape(n * c, 1, h, w)
    x = LUTclip(x)

    batch_s1 = module(x, True)
    batch_s2 = module(torch.rot90(x, 1, [2, 3]), True)
    batch_s2 = torch.rot90(batch_s2, 3, [2, 3])

    batch_s3 = module(torch.rot90(x, 2, [2, 3]), True)
    batch_s3 = torch.rot90(batch_s3, 2, [2, 3])

    batch_s4 = module(torch.rot90(x, 3, [2, 3]), True)
    batch_s4 = torch.rot90(batch_s4, 1, [2, 3])

    batch_s = sum(LUTclip(batch) for batch in [batch_s1, batch_s2, batch_s3, batch_s4])
    return batch_s.reshape(n, c, batch_s.shape[-2], batch_s.shape[-1])
