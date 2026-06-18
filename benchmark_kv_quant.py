"""
End-to-end engine benchmark for comparing FP16 vs quantized KV cache (future).

Usage:
  # FP16 baseline
  uv run python benchmark_engine.py --kv-cache-dtype fp16

  # INT8 KV (after quantization is implemented)
  uv run python benchmark_engine.py --kv-cache-dtype int8

  # Save JSON for later comparison
  uv run python benchmark_engine.py --kv-cache-dtype fp16 --output-json results/fp16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams


DEFAULT_PROMPTS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "give me your opinion on the impact of artificial intelligence on society",
]


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


from myvllm.quantization.kv_cache_quant import kv_cache_pool_bytes
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


def estimate_kv_cache_pool_mb(config: dict) -> float:
        return kv_cache_pool_bytes(
        max_cached_blocks=config["max_cached_blocks"],
        block_size=config["block_size"],
        num_layers=config["num_layers"],
        num_kv_heads=config["num_kv_heads"] // config.get("world_size", 1),
        head_dim=config.get("head_dim", config["hidden_size"] // config["num_heads"]),
        kv_cache_dtype=config.get("kv_cache_dtype", "fp16"),
    ) / 1024**2


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


def print_summary(report: dict) -> None:
    print("\n" + "=" * 72)
    print("Engine Benchmark Summary")
    print("=" * 72)

    meta = report["meta"]
    print(f"kv_cache_dtype     : {meta['kv_cache_dtype']}")
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

    print("\n[Notes]")
    print("- Decode TPS (end-to-end) is the primary metric for quant comparison.")
    print("- Peak GPU memory may stay similar after KV quant because this engine")
    print("  pre-allocates KV cache to fill available GPU memory; compare")
    print("  max_cached_blocks and kv_cache_pool_mb instead.")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end MinivLLM engine benchmark")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["fp16", "bf16", "int8", "fp8"],
        default="fp16",
        help="KV cache dtype. int8/fp8 require future quantization support.",
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
        help="Enable Qwen3 thinking mode in chat template.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs after warmup.")
    parser.add_argument("--verbose", action="store_true", help="Print per-step throughput.")
    parser.add_argument("--output-json", type=str, default="", help="Save report to JSON file.")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict:
    config = build_default_config(
        model_name_or_path=args.model,
        max_model_length=args.max_model_length,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    cache_dir = str(Path(args.cache_dir).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_length=args.max_model_length,
    )
    chat_prompts = build_chat_prompts(tokenizer, DEFAULT_PROMPTS, args.enable_thinking)

    print("Initializing engine (model load + KV cache allocation)...")
    llm = BenchmarkEngine(config=config)
    kv_cache_pool_mb = estimate_kv_cache_pool_mb(llm.config)

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
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "kv_cache_dtype": args.kv_cache_dtype,
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
    }
    return report


def main() -> None:
    args = parse_args()
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
