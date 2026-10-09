from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, Tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    pad_mode = "reflect"
    if (pad_h > 0 and pad_h >= height) or (pad_w > 0 and pad_w >= width):
        pad_mode = "replicate"
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
    return x, (pad_h, pad_w)


def _window_partition(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    x, pad_hw = _pad_to_multiple(x, patch_size)
    batch, channels, height, width = x.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    x = x.view(batch, channels, grid_h, patch_size, grid_w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(batch * grid_h * grid_w, channels, patch_size, patch_size)
    return windows, (grid_h, grid_w), pad_hw


def _window_reverse(
    windows: torch.Tensor,
    batch: int,
    channels: int,
    grid_hw: Tuple[int, int],
    patch_size: int,
    pad_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> torch.Tensor:
    grid_h, grid_w = grid_hw
    x = windows.view(batch, grid_h, grid_w, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch, channels, grid_h * patch_size, grid_w * patch_size)
    pad_h, pad_w = pad_hw
    height, width = orig_hw
    if pad_h > 0:
        x = x[:, :, : height, :]
    if pad_w > 0:
        x = x[:, :, :, : width]
    return x


class MinPSparseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int = 8,
        pbase: float = 0.7,
        trusted_downsample: int = 8,
        attention_impl: str = "legacy",
        trusted_score_impl: str = "legacy",
        sparse_topk: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.pbase = float(pbase)
        self.trusted_downsample = int(trusted_downsample)
        self.attention_impl = str(attention_impl).lower()
        self.trusted_score_impl = str(trusted_score_impl).lower()
        self.sparse_topk = max(1, int(sparse_topk))

        if self.attention_impl not in {"legacy", "optimized"}:
            raise RuntimeError(f"Unsupported attention_impl: {attention_impl}")
        if self.trusted_score_impl not in {"legacy", "fast"}:
            raise RuntimeError(f"Unsupported trusted_score_impl: {trusted_score_impl}")

                                              
                                                                                     
                                                                                     
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=3, padding=1, groups=dim, bias=True)

    def _trusted_scalar_legacy(self, attention_map: torch.Tensor) -> torch.Tensor:
                                                                                           
        pooled = attention_map.unsqueeze(1)
        if pooled.shape[-1] > self.trusted_downsample or pooled.shape[-2] > self.trusted_downsample:
            pooled = F.interpolate(
                pooled,
                size=(self.trusted_downsample, self.trusted_downsample),
                mode="bilinear",
                align_corners=False,
            )
        matrix = pooled.squeeze(1)
        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(matrix.float()).abs()
        rho_hat = torch.sigmoid(eigvals.mean(dim=-1, keepdim=True))
                                                                                    
                                                                            
        return rho_hat.view(-1, 1, 1).to(dtype=attention_map.dtype)

    def _trusted_scalar_fast(self, attention_map: torch.Tensor) -> torch.Tensor:
        pooled = attention_map.unsqueeze(1)
        if pooled.shape[-1] > self.trusted_downsample or pooled.shape[-2] > self.trusted_downsample:
            pooled = F.adaptive_avg_pool2d(pooled, output_size=(self.trusted_downsample, self.trusted_downsample))
        matrix = pooled.squeeze(1)
        diag_strength = matrix.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
        mean_energy = matrix.abs().mean(dim=(-1, -2))
        variance = matrix.var(dim=(-1, -2), unbiased=False)
        score = (diag_strength + mean_energy + variance) / 3.0
        return torch.sigmoid(score).view(-1, 1, 1).to(dtype=attention_map.dtype)

    def _trusted_scalar(self, attention_map: torch.Tensor) -> torch.Tensor:
        if self.trusted_score_impl == "fast":
            return self._trusted_scalar_fast(attention_map)
        return self._trusted_scalar_legacy(attention_map)

    def _compute_attention_map(self, q_tokens: torch.Tensor, k_tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        q_fft = torch.fft.fft(q_tokens.float(), dim=-1, norm="ortho")
        k_fft = torch.fft.fft(k_tokens.float(), dim=-1, norm="ortho")
        attention_freq = q_fft @ k_fft.transpose(-1, -2).conj()
        attention_map = torch.fft.ifft2(attention_freq, norm="ortho").real
        return attention_map.to(dtype=dtype)

    def _forward_legacy(
        self,
        attention_map: torch.Tensor,
        trusted_threshold: torch.Tensor,
        v_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sparse_map = torch.where(attention_map >= trusted_threshold, attention_map, torch.zeros_like(attention_map))
        return sparse_map @ v_tokens

    def _forward_optimized(
        self,
        attention_map: torch.Tensor,
        trusted_threshold: torch.Tensor,
        v_tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_count = attention_map.shape[-1]
        topk = min(self.sparse_topk, token_count)

        topk_scores, topk_indices = torch.topk(attention_map, k=topk, dim=-1, largest=True, sorted=False)
        keep_mask = topk_scores >= trusted_threshold
        keep_mask[..., 0] = True
        topk_scores = torch.where(keep_mask, topk_scores, torch.zeros_like(topk_scores))

        window_ids = torch.arange(v_tokens.shape[0], device=v_tokens.device).view(-1, 1, 1)
        selected_v = v_tokens[window_ids, topk_indices]
        return torch.sum(topk_scores.unsqueeze(-1) * selected_v, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q_windows, grid_hw, pad_hw = _window_partition(q, self.patch_size)
        k_windows, _, _ = _window_partition(k, self.patch_size)
        v_windows, _, _ = _window_partition(v, self.patch_size)

        q_tokens = q_windows.flatten(2).transpose(1, 2).contiguous()
        k_tokens = k_windows.flatten(2).transpose(1, 2).contiguous()
        v_tokens = v_windows.flatten(2).transpose(1, 2).contiguous()

        attention_map = self._compute_attention_map(q_tokens, k_tokens, dtype=x.dtype)

        base_threshold = self.pbase * attention_map.amax(dim=-1, keepdim=True)
        trusted_threshold = base_threshold * self._trusted_scalar(attention_map)
        if self.attention_impl == "optimized":
            out_tokens = self._forward_optimized(attention_map, trusted_threshold, v_tokens)
        else:
            out_tokens = self._forward_legacy(attention_map, trusted_threshold, v_tokens)
        if out_tokens.numel() != v_windows.numel():
            raise RuntimeError(
                "MSA shape mismatch: expected token output to match V windows, "
                f"got out_tokens={tuple(out_tokens.shape)} and v_windows={tuple(v_windows.shape)}."
            )
        out_windows = out_tokens.transpose(1, 2).contiguous().view_as(v_windows)
        out = _window_reverse(
            windows=out_windows,
            batch=batch,
            channels=out_windows.shape[1],
            grid_hw=grid_hw,
            patch_size=self.patch_size,
            pad_hw=pad_hw,
            orig_hw=(height, width),
        )
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(round(dim * float(expansion)))
        self.depthwise = nn.Conv2d(dim, hidden, kernel_size=3, padding=1, groups=dim, bias=True)
        self.pointwise = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.project = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.gelu(self.depthwise(x))
        gate = self.pointwise(x)
        return self.project(feat * gate)


class TrustedSparseBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int = 8,
        pbase: float = 0.7,
        trusted_downsample: int = 8,
        ffn_expansion: float = 2.0,
        attention_impl: str = "legacy",
        trusted_score_impl: str = "legacy",
        sparse_topk: int = 16,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MinPSparseAttention(
            dim=dim,
            patch_size=patch_size,
            pbase=pbase,
            trusted_downsample=trusted_downsample,
            attention_impl=attention_impl,
            trusted_score_impl=trusted_score_impl,
            sparse_topk=sparse_topk,
        )
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim=dim, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FeatureFusionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.enc_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.dec_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.enc_act = nn.PReLU(dim)
        self.dec_act = nn.PReLU(dim)
        self.attn = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=True)

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        if dec_feat.shape[-2:] != enc_feat.shape[-2:]:
            dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[-2:], mode="bilinear", align_corners=False)
        enc_feat_ = self.enc_act(self.enc_proj(enc_feat))
        dec_feat_ = self.dec_act(self.dec_proj(dec_feat))
        matt_x, matt_y = self.attn(torch.cat([enc_feat_, dec_feat_], dim=1)).chunk(2, dim=1)
        matt = torch.softmax(torch.stack([matt_x, matt_y], dim=1), dim=1)
        return matt[:, 0] * enc_feat + matt[:, 1] * dec_feat


class Stage(nn.Module):
    def __init__(
        self,
        dim: int,
        block_count: int,
        patch_size: int = 8,
        pbase: float = 0.7,
        trusted_downsample: int = 8,
        ffn_expansion: float = 2.0,
        attention_impl: str = "legacy",
        trusted_score_impl: str = "legacy",
        sparse_topk: int = 16,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TrustedSparseBlock(
                    dim=dim,
                    patch_size=patch_size,
                    pbase=pbase,
                    trusted_downsample=trusted_downsample,
                    ffn_expansion=ffn_expansion,
                    attention_impl=attention_impl,
                    trusted_score_impl=trusted_score_impl,
                    sparse_topk=sparse_topk,
                )
                for _ in range(block_count)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class TSFormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        stage_blocks: Sequence[int] = (1, 2, 2, 4),
        channel_multipliers: Sequence[int] = (1, 2, 4, 8),
        msa_patch_size: int = 8,
        pbase: float = 0.7,
        trusted_downsample: int = 8,
        ffn_expansion: float = 2.0,
        attention_impl: str = "legacy",
        trusted_score_impl: str = "legacy",
        sparse_topk: int = 16,
    ) -> None:
        super().__init__()
        if len(stage_blocks) != 4:
            raise RuntimeError("stage_blocks must contain exactly four encoder-stage counts.")
        if len(channel_multipliers) != 4:
            raise RuntimeError("channel_multipliers must contain exactly four values.")

        channels = [int(base_channels * scale) for scale in channel_multipliers]
        decoder_blocks = (stage_blocks[2], stage_blocks[1], stage_blocks[0])

        self.model_config = {
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "base_channels": int(base_channels),
            "stage_blocks": tuple(int(v) for v in stage_blocks),
            "channel_multipliers": tuple(int(v) for v in channel_multipliers),
            "msa_patch_size": int(msa_patch_size),
            "pbase": float(pbase),
            "trusted_downsample": int(trusted_downsample),
            "ffn_expansion": float(ffn_expansion),
            "attention_impl": str(attention_impl),
            "trusted_score_impl": str(trusted_score_impl),
            "sparse_topk": int(sparse_topk),
        }

        self.out_channels = int(out_channels)

                                                         
        self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=True)

        self.encoder1 = Stage(
            channels[0],
            stage_blocks[0],
            msa_patch_size,
            pbase,
            trusted_downsample,
            ffn_expansion,
            attention_impl,
            trusted_score_impl,
            sparse_topk,
        )
        self.down1 = Downsample(channels[0], channels[1])
        self.encoder2 = Stage(
            channels[1],
            stage_blocks[1],
            msa_patch_size,
            pbase,
            trusted_downsample,
            ffn_expansion,
            attention_impl,
            trusted_score_impl,
            sparse_topk,
        )
        self.down2 = Downsample(channels[1], channels[2])
        self.encoder3 = Stage(
            channels[2],
            stage_blocks[2],
            msa_patch_size,
            pbase,
            trusted_downsample,
            ffn_expansion,
            attention_impl,
            trusted_score_impl,
            sparse_topk,
        )
        self.down3 = Downsample(channels[2], channels[3])
        self.latent = Stage(
            channels[3],
            stage_blocks[3],
            msa_patch_size,
            pbase,
            trusted_downsample,
            ffn_expansion,
            attention_impl,
            trusted_score_impl,
            sparse_topk,
        )

        self.up3 = Upsample(channels[3], channels[2])
        self.ffb3 = FeatureFusionBlock(channels[2])
        self.decoder3 = Stage(
            channels[2],
            decoder_blocks[0],
            msa_patch_size,
            pbase,
            trusted_downsample,
            ffn_expansion,
            attention_impl,
            trusted_score_impl,
            sparse_topk,
        )

        self.up2 = Upsample(channels[2], channels[1])
        self.ffb2 = FeatureFusionBlock(channels[1])
        self.decoder2 = Stage(
            channels[1],
            decoder_blocks[1],
            msa_patch_size,
            pbase,
            trusted_downsample,
            ffn_expansion,
            attention_impl,
            trusted_score_impl,
            sparse_topk,
        )

        self.up1 = Upsample(channels[1], channels[0])
        self.ffb1 = FeatureFusionBlock(channels[0])
        self.decoder1 = Stage(
            channels[0],
            decoder_blocks[2],
            msa_patch_size,
            pbase,
            trusted_downsample,
            ffn_expansion,
            attention_impl,
            trusted_score_impl,
            sparse_topk,
        )

        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x[:, : self.out_channels]

        x = self.stem(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))
        latent = self.latent(self.down3(enc3))

        dec3 = self.decoder3(self.ffb3(enc3, self.up3(latent)))
        dec2 = self.decoder2(self.ffb2(enc2, self.up2(dec3)))
        dec1 = self.decoder1(self.ffb1(enc1, self.up1(dec2)))

        return self.head(dec1) + residual


def build_tsformer_from_config(config: Dict[str, object]) -> TSFormer:
    return TSFormer(
        in_channels=int(config.get("in_channels", 3)),
        out_channels=int(config.get("out_channels", 3)),
        base_channels=int(config.get("base_channels", 32)),
        stage_blocks=tuple(config.get("stage_blocks", (1, 2, 2, 4))),
        channel_multipliers=tuple(config.get("channel_multipliers", (1, 2, 4, 8))),
        msa_patch_size=int(config.get("msa_patch_size", 8)),
        pbase=float(config.get("pbase", 0.7)),
        trusted_downsample=int(config.get("trusted_downsample", 8)),
        ffn_expansion=float(config.get("ffn_expansion", 2.0)),
        attention_impl=str(config.get("attention_impl", "legacy")),
        trusted_score_impl=str(config.get("trusted_score_impl", "legacy")),
        sparse_topk=int(config.get("sparse_topk", 16)),
    )
