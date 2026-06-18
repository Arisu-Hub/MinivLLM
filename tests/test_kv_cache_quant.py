import torch

from myvllm.quantization.kv_cache_quant import (
    block_bytes,
    dequantize_kv_vector,
    kv_cache_pool_bytes,
    quantize_kv_vector,
    scale_bytes_per_block,
)


BLOCK_KWARGS = dict(
    block_size=256,
    num_layers=28,
    num_kv_heads=8,
    head_dim=128,
)


def test_round_trip_error():
    x = torch.randn(4, 8, 128)
    x_int8, scale = quantize_kv_vector(x)
    x_hat = dequantize_kv_vector(x_int8, scale)

    denom = x.abs().amax().clamp(min=1e-6)
    rel_err = (x_hat - x.float()).abs().max() / denom
    assert rel_err < 0.02


def test_zero_vector():
    x = torch.zeros(8, 128)
    x_int8, scale = quantize_kv_vector(x)
    x_hat = dequantize_kv_vector(x_int8, scale)
    assert torch.allclose(x_hat, x.float(), atol=1e-6)


def test_scale_bytes_same_for_all_dtypes():
    scale_fp16 = scale_bytes_per_block(**{k: v for k, v in BLOCK_KWARGS.items() if k != "head_dim"}, num_kv_heads=8)
    assert scale_fp16 == block_bytes(**BLOCK_KWARGS, kv_cache_dtype="fp16") - (
        BLOCK_KWARGS["block_size"]
        * 2
        * BLOCK_KWARGS["num_layers"]
        * BLOCK_KWARGS["num_kv_heads"]
        * BLOCK_KWARGS["head_dim"]
        * 2
    )


def test_fp16_and_int8_both_include_scale():
    fp16 = block_bytes(**BLOCK_KWARGS, kv_cache_dtype="fp16")
    int8 = block_bytes(**BLOCK_KWARGS, kv_cache_dtype="int8")
    scale_only = scale_bytes_per_block(
        BLOCK_KWARGS["block_size"],
        BLOCK_KWARGS["num_layers"],
        BLOCK_KWARGS["num_kv_heads"],
    )
    assert fp16 > scale_only
    assert int8 > scale_only
    assert int8 < fp16
    assert int8 / fp16 > 0.5


def test_kv_cache_pool_bytes():
    pool = kv_cache_pool_bytes(max_cached_blocks=189, **BLOCK_KWARGS, kv_cache_dtype="fp16")
    assert pool == 189 * block_bytes(**BLOCK_KWARGS, kv_cache_dtype="fp16")
