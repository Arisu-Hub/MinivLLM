"""KV cache quantization utilities.

All cache modes use unified layout: k/v cache tensors + fp32 k_scale/v_scale.
FP16 mode keeps scale at 1.0; INT8 mode writes per-(slot, head) scales on store.

Design doc: docs/kv_quant_design.md
"""

from __future__ import annotations

import torch

KV_CACHE_DTYPES = ("fp32", "fp16", "bf16", "int8", "fp8")

_DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
}


def kv_cache_dtype_bytes(dtype: str) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported kv_cache_dtype: {dtype}. Choose from {KV_CACHE_DTYPES}")
    return _DTYPE_BYTES[dtype]


def resolve_cache_torch_dtype(kv_cache_dtype: str, default_dtype: torch.dtype) -> torch.dtype:
    """Map config kv_cache_dtype string to torch dtype for cache tensors."""
    if kv_cache_dtype == "int8":
        return torch.int8
    if kv_cache_dtype == "fp8":
        return torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else default_dtype
    if kv_cache_dtype == "fp16":
        return torch.float16
    if kv_cache_dtype == "bf16":
        return torch.bfloat16
    if kv_cache_dtype == "fp32":
        return torch.float32
    return default_dtype


def scale_bytes_per_block(block_size: int, num_layers: int, num_kv_heads: int) -> int:
    """Bytes for k_scale + v_scale of one logical block across all layers."""
    return block_size * 2 * num_layers * num_kv_heads * 4


def quantize_kv_vector(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization for the last dimension (head_dim).

    Args:
        x: (..., head_dim) floating tensor

    Returns:
        x_int8: (..., head_dim) int8
        scale:  (...) float32, one scale per head vector
    """
    if x.shape[-1] == 0:
        raise ValueError("head_dim must be > 0")

    x_fp32 = x.float()
    amax = x_fp32.abs().amax(dim=-1)
    scale = torch.clamp(amax / 127.0, min=1e-8)
    x_int8 = torch.round(x_fp32 / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def dequantize_kv_vector(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize cache vectors back to float32 via x_fp = x_cache.to(fp32) * scale."""
    return x_int8.float() * scale.unsqueeze(-1)


def block_bytes(
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: str = "fp16",
) -> int:
    """Bytes for one paged block across all layers (K + V data + scales).

    Scale buffers are always allocated (fp32, init to 1.0) for all dtypes.
    """
    elem_bytes = kv_cache_dtype_bytes(kv_cache_dtype)
    data_bytes = block_size * 2 * num_layers * num_kv_heads * head_dim * elem_bytes
    return data_bytes + scale_bytes_per_block(block_size, num_layers, num_kv_heads)


def kv_cache_pool_bytes(
    max_cached_blocks: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: str = "fp16",
) -> int:
    """Total GPU bytes for the pre-allocated KV cache pool."""
    return max_cached_blocks * block_bytes(
        block_size, num_layers, num_kv_heads, head_dim, kv_cache_dtype
    )
