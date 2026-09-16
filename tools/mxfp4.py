#!/usr/bin/env python3
"""Dequant official DeepSeek MXFP4 expert weights (E2M1 x UE8M0 per 32)."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch


# E2M1 codes 0..7: 0, 0.5, 1, 1.5, 2, 3, 4, 6. High nibble is the second value.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """packed uint8 [..., in/2] -> float32 [..., in] with sign."""
    import torch
    low = packed & 0x0F
    high = packed >> 4
    table = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    mag_lo = table[(low & 7).reshape(-1).long()].reshape(low.shape)
    mag_hi = table[(high & 7).reshape(-1).long()].reshape(high.shape)
    sign_lo = torch.where((low >> 3) != 0, -1.0, 1.0)
    sign_hi = torch.where((high >> 3) != 0, -1.0, 1.0)
    return torch.stack((mag_lo * sign_lo, mag_hi * sign_hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )



def decode_e8m0_byte(value):
    """OCP E8M0: byte zero is subnormal FP32; byte 255 is NaN."""
    if type(value) is not int or not 0 <= value <= 255:
        raise ValueError('E8M0 requires one unsigned byte')
    return float('nan') if value == 255 else 2.0 ** (value - 127)


def dequant_mxfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: int = 32,
) -> torch.Tensor:
    """weight: uint8 [out, in/2] or float4 packed; scale: uint8 [out, in/block] ue8m0.

    Returns bf16 [out, in].
    """
    import torch
    w = weight.view(torch.uint8)
    s = scale.view(torch.uint8)
    vals = unpack_e2m1(w).float()
    out, inn = vals.shape
    if inn % block:
        raise ValueError(f"K={inn} not divisible by block={block}")
    if s.shape != (out, inn // block):
        raise ValueError(f"scale shape {tuple(s.shape)} != {(out, inn // block)}")
    table = torch.tensor([decode_e8m0_byte(b) for b in range(256)],
                         device=s.device, dtype=torch.float32)
    scale_f = table[s.long()]
    vals = vals.view(out, inn // block, block) * scale_f.unsqueeze(-1)
    return vals.reshape(out, inn).to(torch.bfloat16)
