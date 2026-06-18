# INT8 KV Cache 量化设计文档

> 目标：在 MinivLLM 上实现 INT8 KV cache，用于 FP16 vs INT8 的 benchmark 对比。  
> 预计实现周期：第 5–8 周（3 个月计划中的核心开发阶段）。

---

## 1. 背景与动机

### 1.1 当前 KV cache 数据流

```text
Prefill:
  QKV proj → k, v (fp16, 当前 token)
           → flash_attention_prefill(q, k, v)     # 直接用 fp16 k/v 算 attention
           → store_kvcache(k, v → k_cache, v_cache)  # 写入 Paged KV pool

Decode:
  QKV proj → q (fp16, 1 token)
           → paged_attention_decode(q, k_cache, v_cache)  # 从 cache 读历史 K/V
           → store_kvcache(k, v → k_cache, v_cache)       # 写入当前 token
```

### 1.2 当前 cache 布局

| 项目 | 值 |
|------|-----|
| Tensor shape | `(num_blocks, block_size, num_kv_heads, head_dim)` |
| dtype | fp16（与 `default_dtype` 一致） |
| 每层 | 独立的 `k_cache` / `v_cache` |
| 总 pool | `torch.zeros(2, num_layers, max_cached_blocks, ...)` |
| block 数 | 由 `allocate_kv_cache()` 按剩余显存自动计算（当前 baseline: **189 blocks**） |

关键代码：

- 分配：`model_runner.py` → `allocate_kv_cache()`（约 L197–253）
- 写入：`attention.py` → `store_kvcache()` / `store_kvcache_kernel`（约 L7–108）
- 读取：`attention.py` → `paged_attention_decode()` / `paged_attention_decode_kernel`（约 L283–470）
- 调度：`Attention.forward()`（约 L490–536）

### 1.3 量化后的预期收益（本项目的 realistic 预期）

| 指标 | FP16 baseline | INT8 预期 | 说明 |
|------|-------------|-----------|------|
| `max_cached_blocks` | 189 | **~350–380** | block 更小 → 同显存预算下 block 数约翻倍 |
| `kv_cache_pool_mb` | ~5292 | ~5292 | pool 仍占满剩余显存，**总 MB 可能接近** |
| Peak GPU memory | ~13318 MB | ~13318 MB | 同上 |
| Decode TPS (end-to-end) | ~23.5 | ~20–24 | dequant 有开销，可能略降或持平 |
| 生成质量 | baseline | 基本一致 | 需用相同 prompt 肉眼看 + 可选 PPL |

**面试话术**：INT8 KV 的主要收益是 **同等显存预算下可缓存 token 数翻倍**，而非 peak memory 下降。

---

## 2. 量化方案（第一版）

### 2.1 算法：对称 INT8（RTN）

对每个 `(token_slot, kv_head)` 的 `head_dim` 向量独立量化：

```python
scale = max(abs(x)) / 127.0          # x: (head_dim,) fp16/fp32
scale = max(scale, 1e-8)             # 避免除零
x_int8 = round(x / scale).clamp(-128, 127).to(int8)

# 反量化
x_fp = x_int8.to(fp32) * scale
```

选择 **per-(slot, head)** 粒度的原因：

| 粒度 | 优点 | 缺点 |
|------|------|------|
| per-tensor | 实现最简单 | 精度差 |
| **per-(slot, head)** ✅ | 精度与实现复杂度平衡 | 需额外 scale buffer |
| per-channel (per head_dim) | 精度最好 | scale 存储大、kernel 复杂 |

### 2.2 存储布局（统一 scale 设计）

**所有 dtype 共用同一套字段**，避免 Attention / kernel 按 dtype 分支挂载不同结构：

```text
k_cache[layer]: (num_blocks, block_size, num_kv_heads, head_dim)  dtype=fp16 | int8
v_cache[layer]: 同上
k_scale[layer]: (num_blocks, block_size, num_kv_heads)             dtype=fp32, 初始化为 1.0
v_scale[layer]: 同上
```

| 模式 | cache 内容 | scale 行为 | decode 反量化 |
|------|-----------|-----------|--------------|
| **FP16** | 直接存 fp16 | 保持 **1.0**（不写） | `k_fp = k_cache.to(fp32) * 1.0` |
| **INT8** | 存 int8 | **写入时**计算 per-(slot, head) scale | `k_fp = k_cache.to(fp32) * scale` |

统一反量化公式：

```python
k_fp = load(k_cache).to(fp32) * load(k_scale)
v_fp = load(v_cache).to(fp32) * load(v_scale)
```

FP16 模式下 scale 恒为 1.0，等价于直接读 fp16；INT8 模式下 scale 为实际量化因子。

### 2.3 各阶段如何处理

| 阶段 | K/V 来源 | 是否量化 | 做法 |
|------|----------|----------|------|
| Prefill attention | 当前 step 的 fp16 k, v | ❌ 不量化 | 仍走 `flash_attention_prefill` |
| Prefill store | 当前 step 的 fp16 k, v | FP16: 写 fp16；INT8: 量化后写 | 改 `store_kvcache` |
| Decode attention | k_cache + k_scale | 统一 dequant 公式 | 改 `paged_attention_decode_kernel` |
| Decode store | 当前 step 的 fp16 k, v | FP16: 写 fp16；INT8: 量化后写 | 改 `store_kvcache` |

**Prefill 不算 attention 时不读 cache**，因此 prefill 精度不受影响；只有 **store 精度** 和 **decode read 精度** 需要关注。

---

## 3. 模块与接口设计

### 3.1 新增目录

```text
src/myvllm/quantization/
├── __init__.py
└── kv_cache_quant.py    # quantize/dequant 纯函数 + 单元测试友好
```

### 3.2 核心 API（`kv_cache_quant.py`）

```python
KV_CACHE_DTYPES = ("fp16", "int8")

def kv_cache_dtype_bytes(dtype: str) -> int:
    """Return bytes per element for cache tensor."""

def quantize_kv_vector(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: (head_dim,) or (..., head_dim) fp16/fp32
    Returns:
        x_int8: same shape, int8
        scale:  (...,) fp32 — one scale per head vector
    """

def dequantize_kv_vector(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x_int8: (head_dim,) or (..., head_dim) int8
        scale:  (...) fp32
    Returns:
        x_fp: same shape as x_int8, fp16/fp32
    """

def block_bytes(...) -> int:
    """Total bytes for one logical block (K + V data + scales for all dtypes)."""

def kv_cache_pool_bytes(max_cached_blocks, ...) -> int:
    """Total GPU bytes for the pre-allocated KV cache pool."""
```

### 3.3 Config 扩展

在 engine config / `build_default_config()` 中已有：

```python
"kv_cache_dtype": "fp16"  # 或 "int8"
```

`Attention` 模块属性（由 `model_runner.allocate_kv_cache` 注入）：

```python
self.kv_cache_dtype = "fp16"       # 或 "int8"
self.k_cache = ...                 # fp16 或 int8
self.v_cache = ...
self.k_scale = ...                 # 始终 fp32，初始化为 1.0
self.v_scale = ...                 # 始终 fp32，初始化为 1.0
```

行为分支依据 **`kv_cache_dtype` config**，而非 `k_scale.numel() > 0`。

---

## 4. 需要修改的文件清单

| 优先级 | 文件 | 改动 |
|--------|------|------|
| P0 | `quantization/kv_cache_quant.py` | 新建：量化/反量化 + block_bytes |
| P0 | `engine/model_runner.py` | `allocate_kv_cache()` 按 dtype 分配；`block_bytes` 含 scale |
| P0 | `layers/attention.py` | `store_kvcache` 写 int8 + scale；decode kernel 读时 dequant |
| P1 | `layers/attention.py` | `Attention` 持有 k_scale/v_scale；forward 传入 |
| P1 | `benchmark_kv_quant.py` | `--kv-cache-dtype int8` 真正生效，去掉 warning |
| P2 | `tests/test_kv_cache_quant.py` | round-trip 误差、block_bytes 回归测试 |

**不在第一版改动**：`block_manager.py`（逻辑 block 数不变，只是每个 physical block 变小）、`scheduler.py`、prefill flash attention kernel。

---

## 5. 实现步骤（第 5–8 周）

### Step 1：纯函数 + 单元测试（第 5 周）

- [x] 实现 `kv_cache_quant.py`
- [x] 测试：`quantize → dequantize` 最大相对误差 < 1e-2（fp16 下）
- [x] 测试：`block_bytes(fp16)` vs `block_bytes(int8)` ≈ 1:2（含 scale 开销后略小于 2x）

### Step 2：分配路径（第 5–6 周）

- [x] `allocate_kv_cache()` 按 dtype 分配 cache + scale（scale 恒分配，初始 1.0）
- [x] `Attention` 注入 `k_scale` / `v_scale` / `kv_cache_dtype`

### Step 3：写入路径（第 6 周）

- [x] INT8：`quantize_kv_vector` + `store_kvcache_with_scale_kernel`
- [x] FP16：原有 `store_kvcache_kernel`（scale 保持 1.0）

### Step 4：读取路径（第 7 周）

- [x] `paged_attention_decode_kernel` 统一 `cache.to(fp32) * scale`
- [x] `Attention.forward` 传入 scale

### Step 5：打通 benchmark（第 7–8 周）

- [ ] 跑通 FP16 / INT8 benchmark 并保存 JSON

```bash
uv run python benchmark_kv_quant.py --kv-cache-dtype fp16 --output-json results/fp16.json
uv run python benchmark_kv_quant.py --kv-cache-dtype int8 --output-json results/int8.json
```

对比 JSON 中：

- `aggregate.decode_tps`
- `meta.max_cached_blocks`
- `meta.kv_cache_pool_mb`
- 相同 prompt 的生成文本（肉眼看质量）

---

## 6. block_bytes 计算公式

```text
# 每个 logical block 包含 block_size 个 token slot，每层 K+V

scale_bytes = block_size * 2 * num_layers * num_kv_heads * 4   # 所有 dtype 都有

fp16_data = block_size * 2 * num_layers * num_kv_heads * head_dim * 2
fp16_block = fp16_data + scale_bytes

int8_data = block_size * 2 * num_layers * num_kv_heads * head_dim * 1
int8_block = int8_data + scale_bytes

block_bytes = data_bytes + scale_bytes   # scale 不再仅 int8 才有
```

当前 Qwen3-0.6B 参数：`block_size=256, layers=28, kv_heads=8, head_dim=128`

- FP16 每 block（含 scale）：data ~28 MB + scale ~3 MB ≈ **31 MB**
- INT8 每 block（含 scale）：data ~14 MB + scale ~3 MB ≈ **17 MB**

与 baseline 对齐（旧实现未计 scale）：`189 blocks × ~28MB ≈ 5292 MB`  
引入统一 scale 后 FP16 pool 略增（scale 部分），INT8 下 `max_cached_blocks` 仍可提升约 **1.7–1.9x**。

---

## 7. 精度验证计划

### 7.1 单元测试

- 随机向量 round-trip：`max |dequant(quant(x)) - x| / max|x| < 0.01`
- 全零/全相同值边界 case

### 7.2 集成测试

- 同一 prompt、相同 seed/temperature，FP16 vs INT8 输出 diff
- 短 prompt（`introduce yourself`）应高度相似
- 长输出（质数列表）允许末尾 token 略有差异

### 7.3 Benchmark 质量（可选）

- 从生成文本中 strip thinking 块后做字符串相似度
- 有余力：WikiText 子集 PPL（非必须）

---

## 8. 风险与 Fallback

| 风险 | 影响 | 缓解 |
|------|------|------|
| Triton INT8 load+scale 写错 | 输出乱码 / crash | 先用 PyTorch decode fallback 验证 |
| scale 与 cache slot 不对齐 | 精度崩 | store/read 用同一 `(block, offset, head)` 索引 |
| CUDA graph 与 int8 不兼容 | decode 失败 | 保持 `enforce_eager=True`（当前 main 已是） |
| peak 显存不变 | 简历不好写 | 强调 max_cached_blocks 翻倍 |

---

## 9. 第一版明确不做

- FP8 / INT4 KV
- 融合 INT8 GEMM
- 修改 prefill flash attention kernel
- 修改 block_manager / scheduler 逻辑
- 权重量化（W8，留作 stretch goal）

---

## 10. 验收标准（第 8 周末）

- [ ] `--kv-cache-dtype int8` 能完整跑通 `benchmark_kv_quant.py`
- [ ] `max_cached_blocks` 相对 FP16 提升 **≥ 60%**
- [ ] 3 个默认 prompt 输出可读、无明显乱码
- [ ] `results/fp16.json` vs `results/int8.json` 可对比
- [ ] 本文档中的 API 与实现一致（可在 PR 中勾选 checklist）

---

## 附录：数据流图

```mermaid
flowchart TD
    subgraph Prefill
        Q1[Q proj] --> FA[flash_attention_prefill\nfp16 k,v]
        K1[K proj] --> FA
        V1[V proj] --> FA
        K1 --> QNT1[quantize_kv]
        V1 --> QNT1
        QNT1 --> ST1[store_kvcache\nint8 + scale]
    end

    subgraph Decode
        Q2[Q proj 1 token] --> PA[paged_attention_decode]
        KC[(k_cache\nk_scale fp32=1.0|scale)] --> PA
        VC[(v_cache\nv_scale fp32=1.0|scale)] --> PA
        PA --> OUT[output]
        K2[K proj] --> QNT2[quantize_kv]
        V2[V proj] --> QNT2
        QNT2 --> ST2[store_kvcache]
    end
```

---

## 附录：与 benchmark 的对应关系

| benchmark 字段 | FP16 baseline（已测） | INT8 预期 |
|----------------|---------------------|-----------|
| `meta.kv_cache_dtype` | fp16 | int8 |
| `meta.max_cached_blocks` | 189 | ~350+ |
| `aggregate.decode_tps.mean` | 23.51 | 待测 |
| `meta.kv_cache_pool_mb` | 5292.0 | ~5292 |

固定 benchmark 参数：`max_model_length=512, max_tokens=256, runs=3, enable_thinking=False`。
