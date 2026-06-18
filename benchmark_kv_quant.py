"""
End-to-end engine benchmark for comparing FP16 vs INT8 KV cache.

Usage:
  # Throughput + quality samples (auto-fill remaining GPU memory for KV pool)
  uv run python benchmark_kv_quant.py --kv-cache-dtype fp16 --output-json results/fp16.json
  uv run python benchmark_kv_quant.py --kv-cache-dtype int8 --output-json results/int8.json \
      --compare-with results/fp16.json

  # Fixed-block memory comparison (recommended for showing INT8 memory savings)
  # Common presets at block_size=256 (Qwen3-0.6B): 128 (~32K tokens), 256 (~65K), 512 (~131K)
  uv run python benchmark_kv_quant.py --kv-cache-dtype fp16 --memory-test \
      --fixed-max-cached-blocks 256 -o results/fp16_fixed256.json
  uv run python benchmark_kv_quant.py --kv-cache-dtype int8 --memory-test \
      --fixed-max-cached-blocks 256 -o results/int8_fixed256.json \
      --compare-with results/fp16_fixed256.json

  # Long-context stress test (slow; prefer --memory-test for resume demos)
  uv run python benchmark_kv_quant.py --kv-cache-dtype fp16 --long-context-test \
      --output-json results/fp16_long.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine
from myvllm.quantization.kv_cache_quant import (
    kv_bytes_per_token,
    kv_cache_pool_bytes,
    token_capacity_for_blocks,
)
from myvllm.sampling_parameters import SamplingParams


DEFAULT_PROMPTS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "give me your opinion on the impact of artificial intelligence on society",
]

LONG_CONTEXT_USER_PREFIX = (
    "Read the long document below carefully. Reply with exactly one word: OK\n\n"
)
LONG_CONTEXT_FILLER = (
    "Artificial intelligence is transforming science, industry, and daily life. "
    "This sentence repeats with small variations to simulate a very long context window. "
)

# block_size=256 presets (industry-style context lengths: 32K / 65K / 131K tokens)
FIXED_BLOCKS_PRESETS = {
    128: "32,768 tokens (~32K context, common long-doc baseline)",
    256: "65,536 tokens (~65K context, recommended default)",
    512: "131,072 tokens (~131K context, high-context stress)",
}


def build_default_config(**overrides) -> dict:
    config = {
        "max_num_sequences": 16,
        "max_num_batched_tokens": 1024,
        "max_cached_blocks": 1024,
        "block_size": 256,
        "world_size": 1,
        "model_name_or_path": "Qwen/Qwen3-0.6B",
        "enforce_eager": True,
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_heads": 16,
        "head_dim": 128,
        "num_kv_heads": 8,
        "intermediate_size": 3072,
        "num_layers": 28,
        "tie_word_embeddings": True,
        "base": 1000000,
        "rms_norm_epsilon": 1e-6,
        "qkv_bias": False,
        "scale": 1,
        "max_position": 32768,
        "ffn_bias": False,
        "max_num_batch_tokens": 4096,
        "max_model_length": 512,
        "gpu_memory_utilization": 0.9,
        "eos": 151645,
        "kv_cache_dtype": "fp16",
    }
    config.update(overrides)
    return config


@dataclass
class RunStats:
    prefill_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_tokens: int = 0
    decode_time_s: float = 0.0
    decode_step_times_by_batch: dict[int, list[float]] = field(default_factory=dict)
    num_prompts: int = 0
    total_completion_tokens: int = 0
    peak_gpu_memory_mb: float = 0.0

    @property
    def prefill_tps(self) -> float:
        return self.prefill_tokens / self.prefill_time_s if self.prefill_time_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        return self.decode_tokens / self.decode_time_s if self.decode_time_s > 0 else 0.0

    def steady_state_decode_tps(self) -> dict[int, dict[str, float]]:
        result: dict[int, dict[str, float]] = {}
        for batch_size, times in self.decode_step_times_by_batch.items():
            step_tps = [batch_size / t for t in times]
            entry = {"mean_tps": statistics.mean(step_tps), "num_steps": len(times)}
            if len(step_tps) > 1:
                entry["std_tps"] = statistics.stdev(step_tps)
            result[batch_size] = entry
        return result


class BenchmarkEngine(LLMEngine):
    """Collect end-to-end throughput stats during generation."""

    def generate(self, prompts, sampling_params, verbose: bool = False):
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)

        generated_tokens = {}
        prefill_tokens = 0
        prefill_time = 0.0
        decode_tokens = 0
        decode_time = 0.0
        decode_step_times_by_batch: dict[int, list[float]] = {}

        while not self.scheduler.is_finished():
            start_t = time.perf_counter()
            outputs, num_processed_tokens, is_prefill = self.step()
            running_time = time.perf_counter() - start_t + 1e-12

            if is_prefill:
                prefill_tokens += num_processed_tokens
                prefill_time += running_time
                if verbose:
                    print(
                        f"[prefill] batch_tokens={num_processed_tokens} "
                        f"tps={num_processed_tokens / running_time:.2f}"
                    )
            else:
                decode_tokens += num_processed_tokens
                decode_time += running_time
                decode_step_times_by_batch.setdefault(num_processed_tokens, []).append(running_time)
                if verbose:
                    print(
                        f"[decode] batch_size={num_processed_tokens} "
                        f"tps={num_processed_tokens / running_time:.2f}"
                    )

            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        ordered_ids = sorted(generated_tokens.keys())
        token_ids = [generated_tokens[seq_id] for seq_id in ordered_ids]
        self.last_run_stats = RunStats(
            prefill_tokens=prefill_tokens,
            prefill_time_s=prefill_time,
            decode_tokens=decode_tokens,
            decode_time_s=decode_time,
            decode_step_times_by_batch=decode_step_times_by_batch,
            num_prompts=len(prompts),
            total_completion_tokens=sum(len(tokens) for tokens in token_ids),
        )
        return {
            "text": [self.tokenizer.decode(tokens) for tokens in token_ids],
            "token_ids": token_ids,
        }


def kv_head_params(config: dict) -> tuple[int, int]:
    num_kv_heads = config["num_kv_heads"] // config.get("world_size", 1)
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_heads"])
    return num_kv_heads, head_dim


def estimate_kv_cache_pool_mb(config: dict, max_cached_blocks: int | None = None) -> float:
    blocks = max_cached_blocks if max_cached_blocks is not None else config["max_cached_blocks"]
    num_kv_heads, head_dim = kv_head_params(config)
    return kv_cache_pool_bytes(
        max_cached_blocks=blocks,
        block_size=config["block_size"],
        num_layers=config["num_layers"],
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_cache_dtype=config.get("kv_cache_dtype", "fp16"),
    ) / 1024**2


def build_memory_theory(config: dict, max_cached_blocks: int) -> dict:
    num_kv_heads, head_dim = kv_head_params(config)
    kv_dtype = config.get("kv_cache_dtype", "fp16")
    bytes_per_token = kv_bytes_per_token(
        config["num_layers"], num_kv_heads, head_dim, kv_dtype
    )
    fp16_bytes_per_token = kv_bytes_per_token(
        config["num_layers"], num_kv_heads, head_dim, "fp16"
    )
    pool_mb = estimate_kv_cache_pool_mb(config, max_cached_blocks)
    fp16_pool_mb = kv_cache_pool_bytes(
        max_cached_blocks=max_cached_blocks,
        block_size=config["block_size"],
        num_layers=config["num_layers"],
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_cache_dtype="fp16",
    ) / 1024**2
    savings_pct = (
        (1.0 - bytes_per_token / fp16_bytes_per_token) * 100.0
        if fp16_bytes_per_token > 0
        else 0.0
    )
    return {
        "kv_bytes_per_token": bytes_per_token,
        "kv_bytes_per_token_fp16": fp16_bytes_per_token,
        "kv_cache_pool_mb": pool_mb,
        "kv_cache_pool_mb_fp16_ref": fp16_pool_mb,
        "kv_pool_savings_vs_fp16_pct": savings_pct,
        "max_token_capacity": token_capacity_for_blocks(max_cached_blocks, config["block_size"]),
    }


def strip_thinking(text: str) -> str:
    tag_pairs = (
        ("think",),
        ("redacted_thinking",),
    )
    for tag in tag_pairs:
        open_tag = f"<{tag[0]}>"
        close_tag = f"</{tag[0]}>"
        pattern = re.escape(open_tag) + r"[\s\S]*?" + re.escape(close_tag)
        text = re.sub(pattern, "", text, count=1)
    return text.strip()


def text_similarity(reference: str, candidate: str) -> float:
    ref = strip_thinking(reference)
    cand = strip_thinking(candidate)
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    return SequenceMatcher(None, ref, cand).ratio()


def build_chat_prompts(tokenizer, user_prompts: list[str], enable_thinking: bool) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for prompt in user_prompts
    ]


def build_long_user_content(tokenizer, target_prompt_tokens: int) -> tuple[str, int]:
    """Build a user message whose token count is close to target_prompt_tokens."""
    content = LONG_CONTEXT_USER_PREFIX
    variant = 0
    while True:
        token_ids = tokenizer.encode(content, add_special_tokens=False)
        if len(token_ids) >= target_prompt_tokens:
            break
        content += f"[seg-{variant}] " + LONG_CONTEXT_FILLER
        variant += 1

    token_ids = token_ids[:target_prompt_tokens]
    content = tokenizer.decode(token_ids, skip_special_tokens=True)
    return content, len(token_ids)


def blocks_needed_for_tokens(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size) if num_tokens > 0 else 0


def aggregate_runs(runs: list[RunStats]) -> dict:
    def agg(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        result = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
        if len(values) > 1:
            result["std"] = statistics.stdev(values)
        else:
            result["std"] = 0.0
        return result

    return {
        "prefill_tps": agg([r.prefill_tps for r in runs]),
        "decode_tps": agg([r.decode_tps for r in runs]),
        "peak_gpu_memory_mb": agg([r.peak_gpu_memory_mb for r in runs]),
        "total_completion_tokens": agg([float(r.total_completion_tokens) for r in runs]),
    }


def run_quality_evaluation(
    llm: BenchmarkEngine,
    tokenizer,
    args: argparse.Namespace,
) -> dict:
    """Run one generation pass for quality inspection (reuses existing engine)."""
    quality_temp = args.quality_temperature if args.quality_temperature is not None else args.temperature
    print(f"\n--- quality evaluation (temperature={quality_temp}) ---")

    chat_prompts = build_chat_prompts(tokenizer, DEFAULT_PROMPTS, args.enable_thinking)
    sampling_params = SamplingParams(
        temperature=quality_temp,
        max_tokens=args.max_tokens,
        max_model_length=args.max_model_length,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    outputs = llm.generate(chat_prompts, sampling_params, verbose=False)
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )

    samples = []
    for idx, (user_prompt, completion) in enumerate(zip(DEFAULT_PROMPTS, outputs["text"])):
        stripped = strip_thinking(completion)
        samples.append(
            {
                "prompt_id": idx,
                "user_prompt": user_prompt,
                "completion": completion,
                "completion_stripped": stripped,
                "completion_chars": len(stripped),
                "completion_tokens": len(outputs["token_ids"][idx]),
            }
        )
        preview = stripped[:200].replace("\n", " ")
        print(f"  [{idx}] tokens={len(outputs['token_ids'][idx]):4d} preview: {preview}")

    return {
        "temperature": quality_temp,
        "max_tokens": args.max_tokens,
        "peak_gpu_memory_mb": peak_mb,
        "samples": samples,
    }


def compare_quality_with_reference(quality: dict, reference_path: Path) -> dict | None:
    if not reference_path.exists():
        print(f"Warning: reference file not found: {reference_path}")
        return None

    with reference_path.open(encoding="utf-8") as f:
        reference_report = json.load(f)

    ref_samples = reference_report.get("quality", {}).get("samples", [])
    if not ref_samples:
        print(f"Warning: no quality.samples in {reference_path}")
        return None

    ref_by_id = {sample["prompt_id"]: sample for sample in ref_samples}
    comparisons = []
    similarities = []

    for sample in quality["samples"]:
        ref = ref_by_id.get(sample["prompt_id"])
        if ref is None:
            continue
        ref_stripped = ref.get("completion_stripped") or strip_thinking(ref["completion"])
        cand_stripped = sample["completion_stripped"]
        sim = text_similarity(ref_stripped, cand_stripped)
        exact_match = ref_stripped == cand_stripped
        length_delta = sample["completion_tokens"] - ref["completion_tokens"]
        comparisons.append(
            {
                "prompt_id": sample["prompt_id"],
                "user_prompt": sample["user_prompt"],
                "similarity": sim,
                "exact_match": exact_match,
                "reference_completion_tokens": ref["completion_tokens"],
                "candidate_completion_tokens": sample["completion_tokens"],
                "completion_token_delta": length_delta,
                "reference_preview": ref_stripped[:300],
                "candidate_preview": cand_stripped[:300],
            }
        )
        similarities.append(sim)

    exact_matches = sum(1 for c in comparisons if c["exact_match"])
    summary = {
        "reference_json": str(reference_path),
        "reference_kv_cache_dtype": reference_report.get("meta", {}).get("kv_cache_dtype"),
        "mean_similarity": statistics.mean(similarities) if similarities else 0.0,
        "min_similarity": min(similarities) if similarities else 0.0,
        "exact_match_count": exact_matches,
        "total_compared": len(comparisons),
        "comparisons": comparisons,
    }
    return summary


def compare_memory_with_reference(memory: dict, reference_path: Path) -> dict | None:
    if not reference_path.exists():
        print(f"Warning: reference file not found: {reference_path}")
        return None

    with reference_path.open(encoding="utf-8") as f:
        reference_report = json.load(f)

    ref_meta = reference_report.get("meta", {})
    ref_memory = reference_report.get("memory", {})
    ref_theory = ref_memory.get("theory", {})
    cand_theory = memory.get("theory", {})

    ref_pool = ref_meta.get("kv_cache_pool_mb") or ref_theory.get("kv_cache_pool_mb")
    cand_pool = memory.get("kv_cache_pool_mb") or cand_theory.get("kv_cache_pool_mb")
    ref_peak = ref_memory.get("peak_gpu_memory_mb")
    cand_peak = memory.get("peak_gpu_memory_mb")

    if ref_pool is None or cand_pool is None:
        print(f"Warning: missing kv_cache_pool_mb in reference {reference_path}")
        return None

    summary = {
        "reference_json": str(reference_path),
        "reference_kv_cache_dtype": ref_meta.get("kv_cache_dtype"),
        "reference_fixed_max_cached_blocks": ref_meta.get("fixed_max_cached_blocks"),
        "kv_cache_pool_mb_delta": cand_pool - ref_pool,
        "kv_cache_pool_mb_savings_pct": (1.0 - cand_pool / ref_pool) * 100.0 if ref_pool else 0.0,
    }
    if ref_peak is not None and cand_peak is not None:
        summary["peak_gpu_memory_mb_delta"] = cand_peak - ref_peak
        summary["peak_gpu_memory_mb_savings_pct"] = (
            (1.0 - cand_peak / ref_peak) * 100.0 if ref_peak else 0.0
        )
    return summary


def run_fixed_blocks_memory_test(args: argparse.Namespace) -> dict:
    blocks = args.fixed_max_cached_blocks or 256
    preset_hint = FIXED_BLOCKS_PRESETS.get(blocks, "custom block count")

    config = build_default_config(
        model_name_or_path=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        fixed_max_cached_blocks=blocks,
        max_model_length=args.max_model_length,
    )

    cache_dir = str(Path(args.cache_dir).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    chat_prompts = build_chat_prompts(tokenizer, DEFAULT_PROMPTS, args.enable_thinking)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=min(args.max_tokens, 64),
        max_model_length=args.max_model_length,
    )

    print("Initializing engine for fixed-block memory test...")
    print(
        f"fixed_max_cached_blocks={blocks} ({preset_hint}), "
        f"block_size={config['block_size']}, kv_cache_dtype={args.kv_cache_dtype}"
    )
    print(
        "Industry-style presets at block_size=256: "
        "128 (~32K tokens), 256 (~65K), 512 (~131K). "
        "INT8 typically saves ~50% KV pool vs FP16 at the same block count."
    )

    llm = BenchmarkEngine(config=config)
    max_cached_blocks = llm.config["max_cached_blocks"]
    theory = build_memory_theory(llm.config, max_cached_blocks)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_t = time.perf_counter()
    llm.generate(chat_prompts, sampling_params, verbose=False)
    elapsed_s = time.perf_counter() - start_t
    stats = llm.last_run_stats

    peak_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    memory = {
        "fixed_max_cached_blocks": max_cached_blocks,
        "kv_cache_pool_mb": estimate_kv_cache_pool_mb(llm.config),
        "peak_gpu_memory_mb": peak_mb,
        "elapsed_s": elapsed_s,
        "decode_tps": stats.decode_tps,
        "prefill_tps": stats.prefill_tps,
        "total_completion_tokens": stats.total_completion_tokens,
        "theory": theory,
    }

    memory_compare = None
    if args.compare_with:
        memory_compare = compare_memory_with_reference(memory, Path(args.compare_with))

    return {
        "meta": {
            "benchmark_type": "fixed_blocks_memory",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "kv_cache_dtype": args.kv_cache_dtype,
            "fixed_max_cached_blocks": max_cached_blocks,
            "max_cached_blocks": max_cached_blocks,
            "block_size": config["block_size"],
            "kv_cache_pool_mb": memory["kv_cache_pool_mb"],
            "max_model_length": args.max_model_length,
            "max_tokens": sampling_params.max_tokens,
            "preset_hint": preset_hint,
        },
        "memory": memory,
        "memory_compare": memory_compare,
    }


def run_long_context_test(args: argparse.Namespace) -> dict:
    block_size = 256
    target_blocks = args.long_context_target_blocks
    target_prompt_tokens = target_blocks * block_size
    max_model_length = target_prompt_tokens + args.long_context_max_tokens + 64
    max_position = max(max_model_length + 4096, 131072)

    config = build_default_config(
        model_name_or_path=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        block_size=block_size,
        max_model_length=max_model_length,
        max_position=max_position,
        max_num_batched_tokens=max_model_length + 128,
        max_num_batch_tokens=max_model_length + 128,
        warmup_max_tokens=1024,
        max_num_sequences=1,
    )

    cache_dir = str(Path(args.cache_dir).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    user_content, user_token_count = build_long_user_content(tokenizer, target_prompt_tokens)
    chat_prompt = build_chat_prompts(tokenizer, [user_content], enable_thinking=False)[0]
    prompt_token_count = len(tokenizer.encode(chat_prompt))

    print("Initializing engine for long-context test...")
    print(
        f"Target blocks={target_blocks}, user_tokens≈{user_token_count}, "
        f"chat_prompt_tokens={prompt_token_count}, max_model_length={max_model_length}"
    )

    llm = BenchmarkEngine(config=config)
    max_cached_blocks = llm.config["max_cached_blocks"]
    blocks_needed = blocks_needed_for_tokens(
        prompt_token_count + args.long_context_max_tokens, block_size
    )

    preflight = {
        "prompt_token_count": prompt_token_count,
        "blocks_needed_estimate": blocks_needed,
        "max_cached_blocks": max_cached_blocks,
        "fits_in_cache_pool": blocks_needed <= max_cached_blocks,
    }

    print(
        f"max_cached_blocks={max_cached_blocks}, "
        f"estimated_blocks_needed={blocks_needed}, "
        f"fits={preflight['fits_in_cache_pool']}"
    )

    result = {
        "success": False,
        "error": None,
        "completion": None,
        "completion_stripped": None,
        "completion_tokens": 0,
        "preflight": preflight,
        "peak_gpu_memory_mb": 0.0,
        "elapsed_s": 0.0,
    }

    if not preflight["fits_in_cache_pool"]:
        result["error"] = (
            f"Preflight failed: need ~{blocks_needed} blocks but pool has "
            f"{max_cached_blocks} blocks."
        )
        print(f"Long-context test skipped: {result['error']}")
        return {
            "meta": {
                "benchmark_type": "long_context",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "kv_cache_dtype": args.kv_cache_dtype,
                "max_cached_blocks": max_cached_blocks,
                "kv_cache_pool_mb": estimate_kv_cache_pool_mb(llm.config),
                "long_context_target_blocks": target_blocks,
                "long_context_max_tokens": args.long_context_max_tokens,
                "max_model_length": max_model_length,
                "prompt_token_count": prompt_token_count,
            },
            "long_context": result,
        }

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.long_context_max_tokens,
        max_model_length=max_model_length,
    )

    start_t = time.perf_counter()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        outputs = llm.generate([chat_prompt], sampling_params, verbose=False)
        completion = outputs["text"][0]
        result["success"] = True
        result["completion"] = completion
        result["completion_stripped"] = strip_thinking(completion)
        result["completion_tokens"] = len(outputs["token_ids"][0])
        result["elapsed_s"] = time.perf_counter() - start_t
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f"Long-context test succeeded in {result['elapsed_s']:.1f}s, "
            f"completion_tokens={result['completion_tokens']}, "
            f"output={result['completion_stripped'][:80]!r}"
        )
    except torch.cuda.OutOfMemoryError as exc:
        result["error"] = f"CUDA OOM: {exc}"
        result["elapsed_s"] = time.perf_counter() - start_t
        print(f"Long-context test failed: {result['error']}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_s"] = time.perf_counter() - start_t
        print(f"Long-context test failed: {result['error']}")

    return {
        "meta": {
            "benchmark_type": "long_context",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "kv_cache_dtype": args.kv_cache_dtype,
            "max_cached_blocks": max_cached_blocks,
            "kv_cache_pool_mb": estimate_kv_cache_pool_mb(llm.config),
            "long_context_target_blocks": target_blocks,
            "long_context_max_tokens": args.long_context_max_tokens,
            "max_model_length": max_model_length,
            "prompt_token_count": prompt_token_count,
        },
        "long_context": result,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    config_overrides = {
        "model_name_or_path": args.model,
        "max_model_length": args.max_model_length,
        "kv_cache_dtype": args.kv_cache_dtype,
    }
    if args.fixed_max_cached_blocks is not None:
        config_overrides["fixed_max_cached_blocks"] = args.fixed_max_cached_blocks

    config = build_default_config(**config_overrides)
    cache_dir = str(Path(args.cache_dir).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_length=args.max_model_length,
    )
    chat_prompts = build_chat_prompts(tokenizer, DEFAULT_PROMPTS, args.enable_thinking)

    print("Initializing engine (model load + KV cache allocation)...")
    if args.fixed_max_cached_blocks is not None:
        print(f"Using fixed_max_cached_blocks={args.fixed_max_cached_blocks}")
    llm = BenchmarkEngine(config=config)
    kv_cache_pool_mb = estimate_kv_cache_pool_mb(llm.config)

    # Quality run first while scheduler is clean (cannot init a second engine: NCCL PG).
    quality = run_quality_evaluation(llm, tokenizer, args)
    quality_compare = None
    if args.compare_with:
        quality_compare = compare_quality_with_reference(quality, Path(args.compare_with))

    measured_runs: list[RunStats] = []
    total_runs = args.warmup_runs + args.runs

    for run_idx in range(total_runs):
        is_warmup = run_idx < args.warmup_runs
        label = "warmup" if is_warmup else f"run {run_idx - args.warmup_runs + 1}/{args.runs}"
        print(f"\n--- {label} ---")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        llm.generate(chat_prompts, sampling_params, verbose=args.verbose)
        stats = llm.last_run_stats

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

        if is_warmup:
            print(
                f"warmup decode_tps={stats.decode_tps:.2f}, "
                f"peak_mem={stats.peak_gpu_memory_mb:.1f} MB"
            )
            continue

        measured_runs.append(stats)
        print(
            f"decode_tps={stats.decode_tps:.2f}, prefill_tps={stats.prefill_tps:.2f}, "
            f"completion_tokens={stats.total_completion_tokens}, "
            f"peak_mem={stats.peak_gpu_memory_mb:.1f} MB"
        )

    report = {
        "meta": {
            "benchmark_type": "throughput",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "kv_cache_dtype": args.kv_cache_dtype,
            "fixed_max_cached_blocks": args.fixed_max_cached_blocks,
            "max_cached_blocks": llm.config["max_cached_blocks"],
            "kv_cache_pool_mb": kv_cache_pool_mb,
            "block_size": config["block_size"],
            "max_model_length": args.max_model_length,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "enable_thinking": args.enable_thinking,
            "num_prompts": len(chat_prompts),
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
        },
        "aggregate": aggregate_runs(measured_runs),
        "runs": [asdict(run) for run in measured_runs],
        "quality": quality,
    }
    if quality_compare is not None:
        report["quality_compare"] = quality_compare
    return report


def print_summary(report: dict) -> None:
    meta = report["meta"]
    benchmark_type = meta.get("benchmark_type", "throughput")

    if benchmark_type == "long_context":
        print_long_context_summary(report)
        return

    if benchmark_type == "fixed_blocks_memory":
        print_fixed_blocks_memory_summary(report)
        return

    print("\n" + "=" * 72)
    print("Engine Benchmark Summary")
    print("=" * 72)
    print(f"kv_cache_dtype     : {meta['kv_cache_dtype']}")
    if meta.get("fixed_max_cached_blocks") is not None:
        print(f"fixed_max_blocks   : {meta['fixed_max_cached_blocks']}")
    print(f"max_cached_blocks  : {meta['max_cached_blocks']}")
    print(f"kv_cache_pool_mb   : {meta['kv_cache_pool_mb']:.1f}")
    print(f"max_model_length   : {meta['max_model_length']}")
    print(f"num_prompts        : {meta['num_prompts']}")
    print(f"warmup_runs        : {meta['warmup_runs']}")
    print(f"measured_runs      : {meta['measured_runs']}")

    agg = report["aggregate"]
    prefill = agg["prefill_tps"]
    decode = agg["decode_tps"]
    memory = agg["peak_gpu_memory_mb"]

    print("\n[Primary metrics]")
    print(
        f"Decode TPS (end-to-end): {decode['mean']:.2f} ± {decode['std']:.2f} "
        f"(min={decode['min']:.2f}, max={decode['max']:.2f})"
    )
    print(
        f"Prefill TPS             : {prefill['mean']:.2f} ± {prefill['std']:.2f} "
        f"(min={prefill['min']:.2f}, max={prefill['max']:.2f})"
    )
    print(
        f"Peak GPU memory         : {memory['mean']:.1f} ± {memory['std']:.1f} MB "
        f"(min={memory['min']:.1f}, max={memory['max']:.1f})"
    )

    if report["runs"]:
        last_run = report["runs"][-1]
        steady = RunStats(**last_run).steady_state_decode_tps()
        if steady:
            print("\n[Steady-state decode TPS, last run, grouped by batch size]")
            for batch_size in sorted(steady.keys(), reverse=True):
                item = steady[batch_size]
                if "std_tps" in item:
                    print(
                        f"  batch={batch_size}: {item['mean_tps']:.2f} ± "
                        f"{item['std_tps']:.2f} tok/s ({item['num_steps']} steps)"
                    )
                else:
                    print(
                        f"  batch={batch_size}: {item['mean_tps']:.2f} tok/s "
                        f"({item['num_steps']} steps)"
                    )

    quality = report.get("quality", {})
    if quality.get("samples"):
        temp = quality.get("temperature", "?")
        print(f"\n[Generation quality samples (temperature={temp}, thinking stripped)]")
        for sample in quality["samples"]:
            preview = sample["completion_stripped"][:200].replace("\n", " ")
            print(
                f"  [{sample['prompt_id']}] tokens={sample['completion_tokens']:4d} "
                f"prompt={sample['user_prompt'][:40]!r}"
            )
            print(f"       output: {preview}")

    quality_compare = report.get("quality_compare")
    if quality_compare:
        ref_dtype = quality_compare["reference_kv_cache_dtype"]
        cand_dtype = report["meta"]["kv_cache_dtype"]
        print(f"\n[Quality vs reference: {ref_dtype} -> {cand_dtype}]")
        print(f"  reference file      : {quality_compare['reference_json']}")
        print(
            f"  mean similarity     : {quality_compare['mean_similarity']:.4f} "
            f"(1.0 = text identical after stripping thinking blocks)"
        )
        print(
            f"  exact match         : "
            f"{quality_compare['exact_match_count']}/{quality_compare['total_compared']} prompts"
        )
        for item in quality_compare["comparisons"]:
            match_label = "SAME" if item["exact_match"] else "DIFF"
            print(
                f"\n  [{item['prompt_id']}] {match_label}  sim={item['similarity']:.4f}  "
                f"token_delta={item['completion_token_delta']:+d}  "
                f"prompt={item['user_prompt'][:35]!r}"
            )
            print(f"    FP16: {item['reference_preview'][:200]}")
            print(f"    {cand_dtype.upper():4s}: {item['candidate_preview'][:200]}")

    print("\n[Notes]")
    print("- Decode TPS (end-to-end) is the primary metric for quant comparison.")
    print("- With auto-fill KV pool, peak GPU memory stays similar; INT8 wins on")
    print("  max_cached_blocks. Use --memory-test --fixed-max-cached-blocks 256")
    print("  to show lower peak memory at the same block count.")
    print("- Common fixed-block presets (block_size=256): 128/256/512 blocks.")
    print("=" * 72)


def print_fixed_blocks_memory_summary(report: dict) -> None:
    meta = report["meta"]
    memory = report["memory"]
    theory = memory["theory"]

    print("\n" + "=" * 72)
    print("Fixed-Block Memory Comparison Summary")
    print("=" * 72)
    print(f"kv_cache_dtype              : {meta['kv_cache_dtype']}")
    print(f"fixed_max_cached_blocks     : {meta['fixed_max_cached_blocks']}")
    print(f"preset                      : {meta.get('preset_hint', '')}")
    print(f"block_size                  : {meta['block_size']}")
    print(f"max_token_capacity          : {theory['max_token_capacity']:,} tokens")
    print(f"kv_bytes_per_token          : {theory['kv_bytes_per_token']:.1f} B")
    print(f"kv_bytes_per_token (fp16)   : {theory['kv_bytes_per_token_fp16']:.1f} B")
    print(f"kv_cache_pool_mb            : {memory['kv_cache_pool_mb']:.1f}")
    print(f"kv_cache_pool_mb (fp16 ref) : {theory['kv_cache_pool_mb_fp16_ref']:.1f}")
    print(f"theoretical KV savings      : {theory['kv_pool_savings_vs_fp16_pct']:.1f}%")
    print(f"peak_gpu_memory_mb          : {memory['peak_gpu_memory_mb']:.1f}")
    print(f"short generate elapsed_s    : {memory['elapsed_s']:.1f}")
    print(f"decode_tps (smoke test)     : {memory['decode_tps']:.2f}")

    memory_compare = report.get("memory_compare")
    if memory_compare:
        ref_dtype = memory_compare["reference_kv_cache_dtype"]
        cand_dtype = meta["kv_cache_dtype"]
        print(f"\n[Memory vs reference: {ref_dtype} -> {cand_dtype}]")
        print(f"  reference file           : {memory_compare['reference_json']}")
        print(
            f"  kv_cache_pool_mb delta   : {memory_compare['kv_cache_pool_mb_delta']:+.1f} MB "
            f"({memory_compare['kv_cache_pool_mb_savings_pct']:.1f}% savings vs ref pool)"
        )
        if "peak_gpu_memory_mb_delta" in memory_compare:
            print(
                f"  peak_gpu_memory_mb delta : "
                f"{memory_compare['peak_gpu_memory_mb_delta']:+.1f} MB "
                f"({memory_compare['peak_gpu_memory_mb_savings_pct']:.1f}% savings vs ref peak)"
            )

    print("\n[Recommended presets at block_size=256]")
    for blocks, hint in FIXED_BLOCKS_PRESETS.items():
        print(f"  --fixed-max-cached-blocks {blocks}: {hint}")
    print("=" * 72)


def print_long_context_summary(report: dict) -> None:
    meta = report["meta"]
    result = report["long_context"]
    preflight = result["preflight"]

    print("\n" + "=" * 72)
    print("Long-Context Stress Test Summary")
    print("=" * 72)
    print(f"kv_cache_dtype          : {meta['kv_cache_dtype']}")
    print(f"max_cached_blocks       : {meta['max_cached_blocks']}")
    print(f"target_blocks           : {meta['long_context_target_blocks']}")
    prompt_tokens = preflight.get("prompt_token_count", meta.get("prompt_token_count"))
    if prompt_tokens is not None:
        print(f"prompt_token_count      : {prompt_tokens}")
    print(f"estimated_blocks_needed : {preflight['blocks_needed_estimate']}")
    print(f"fits_in_cache_pool      : {preflight['fits_in_cache_pool']}")
    print(f"success                 : {result['success']}")
    if result["error"]:
        print(f"error                   : {result['error']}")
    if result["success"]:
        print(f"elapsed_s               : {result['elapsed_s']:.1f}")
        print(f"completion_tokens       : {result['completion_tokens']}")
        print(f"completion_preview      : {result['completion_stripped'][:120]!r}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end MinivLLM engine benchmark")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["fp16", "bf16", "int8", "fp8"],
        default="fp16",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--max-model-length", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "-o", "--output-json",
        type=str,
        default="",
        help="Save report to JSON file.",
    )
    parser.add_argument(
        "--quality-temperature",
        type=float,
        default=None,
        help="Temperature for quality evaluation (default: same as --temperature).",
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default="",
        help="Compare quality.samples against a reference JSON (e.g. results/fp16.json).",
    )
    parser.add_argument(
        "--fixed-max-cached-blocks",
        type=int,
        default=None,
        help="Fix KV cache pool size (blocks). At block_size=256: "
             "128 (~32K tokens), 256 (~65K, default for --memory-test), 512 (~131K).",
    )
    parser.add_argument(
        "--memory-test",
        action="store_true",
        help="Fast fixed-block memory comparison (short generate, skips throughput runs).",
    )
    parser.add_argument(
        "--long-context-test",
        action="store_true",
        help="Run isolated long-context block-capacity test only (skips throughput runs).",
    )
    parser.add_argument(
        "--long-context-target-blocks",
        type=int,
        default=400,
        help="Target logical blocks for long prompt (~400 exceeds typical FP16 ~373).",
    )
    parser.add_argument(
        "--long-context-max-tokens",
        type=int,
        default=8,
        help="Max new tokens for long-context test generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.memory_test:
        if args.fixed_max_cached_blocks is None:
            args.fixed_max_cached_blocks = 256
        report = run_fixed_blocks_memory_test(args)
    elif args.long_context_test:
        report = run_long_context_test(args)
    else:
        report = run_benchmark(args)

    print_summary(report)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
