# Changelog

All notable changes to `kv-planner` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — Correctness release

This release rewrites the physics engine. Every recommendation the library
emits (batch size, quantization, GPU choice, cost per million tokens)
depends on formulas that were wrong in 0.1.x. **If you saved plans under
0.1.x, re-run them** — the numbers will change, and the new ones match
published literature.

### Fixed — physics

- **FLOPs per token now includes the attention term** (`4·d·s`) that was
  missing in 0.1.x. The old `24 · n_layers · d_model²` shortcut under-
  predicted long-context decode by a factor that grows linearly with
  context length. FLOPs also now accounts for GQA-smaller K/V projections
  and the SwiGLU 3-matmul FFN. Sources: kipply's "Transformer Inference
  Arithmetic", PaLM paper, EleutherAI cookbook.
- **Decode latency now reads the entire growing KV cache**, not one
  token's worth. Pre-fix, the per-step decode latency was independent
  of context length — physically wrong. Sources: kipply, vLLM docs.
- **Memory-bound prefill latency now scales with batch × seq** via an
  explicit KV-write term. Pre-fix it returned a constant equal to the
  one-time weight-load cost.
- **Tensor-parallel AllReduce cost** now uses the correct
  ring-AllReduce formula `2·(N−1)/N · M/B + 2·(N−1)·α`, with **two**
  AllReduces per layer (Megatron-LM post-attention + post-MLP). Pre-fix
  the code scaled linearly in TP and counted only one AllReduce per
  layer.
- **`model.total_params()` is now GQA-aware.** Pre-fix it counted Q, K,
  V, and O as all `d × d`; post-fix, K and V are `d × (kv_h · d_h)`.
  Llama-3 8B now reports 8.03 B (matches Meta's published number);
  pre-fix it reported 6.97 B.
- **Fragmentation is counted exactly once.** Pre-fix, `paged.py`
  multiplied by `(1 + 0.04)`, then `hardware.py` multiplied by
  `PAGED_ATTENTION_EFFICIENCY = 0.96` — a triple-counted overhead on top
  of the ceiling-block allocation that already encodes the only source
  of internal fragmentation. Kept only the ceiling; the vLLM-reported
  <4 % figure is the total.
- **Arithmetic-intensity derivations are now separate** for prefill and
  decode. Decode no longer adds activation memory (FlashAttention
  streams activations — Dao et al. arxiv 2205.14135).
- **Peak TFLOPS is precision-aware.** Pre-fix, Roofline on an FP8 or
  INT8 workload compared AI against the FP16 ridge point; post-fix,
  `peak_tflops_for(precision)` returns the correct peak for each dtype.
- **Precision → bytes table is a single source of truth** in
  `domain/precision.py`. Pre-fix it was duplicated across 6 files and
  had already diverged slightly.

### Removed — physically incorrect features

- **Deleted `HardwareSpec.kv_cache_super_linear_scaling_factor()`.** It
  returned 13.9× for TP=2 and 28× for TP=8 — nonsensical values that
  caused `max_batch_size` to over-predict by ~7× and silently produce
  out-of-memory errors at runtime. Memory scales linearly with TP; the
  effect that the method was trying to capture (sharding weights frees
  per-GPU memory for the KV cache) is now handled correctly by
  `available_kv_cache_memory_gb(model_size_gb)`.
- **Removed `PAGED_ATTENTION_EFFICIENCY = 0.96`** from
  `available_kv_cache_memory_gb()`. Fragmentation is already in the
  block-ceiling math; this was a duplicate penalty.
- **Removed `FRAGMENTATION_OVERHEAD = 0.04`** multiplier from
  `PagedMemoryCalculator`. Same double-count.
- **Removed `VALIDATED_RTX_5060_LAPTOP` profile** (7.2 % retention
  factor) from `laptop_adjustments`. Derivation was cross-tier
  (laptop RTX 5060 vs implicit desktop RTX 5090), not a real
  laptop-vs-same-tier-desktop penalty. Real same-tier retention is
  **50–65 %** per reviewers. The 255.63 tok/s measurement on the
  RTX 5060 Laptop is preserved as a test fixture but no longer used as
  a retention ratio.

### Changed — hardware database

Every GPU's `peak_tflops_fp16` value has been updated to the **dense FP16
tensor-core** figure from the vendor whitepaper (not the FP32 CUDA core
number, not the 2:4-sparsity number). Structured sparsity requires
2:4-pruned weights that almost no deployed LLM ships with.

Key corrections:

| GPU | Old (0.1.x) | New (0.2.0) | Source |
|---|---|---|---|
| RTX-5090 | 104.8 | **838** | NVIDIA Blackwell whitepaper |
| RTX-5080 | 56.3 | **450** | NVIDIA Blackwell whitepaper |
| RTX-5070-Ti | 44.0 | **290** | NVIDIA Blackwell whitepaper |
| RTX-5070 | 31.0 | **180** | NVIDIA Blackwell whitepaper |
| RTX-4090 | 82.58 | **165.2** | NVIDIA Ada whitepaper |
| RTX-3090-Ti | 40 | **160** | NVIDIA Ampere GA102 whitepaper |
| RTX-3090 | 35.58 | **142** | NVIDIA Ampere GA102 whitepaper |

New GPUs added: H200-SXM-141GB, H100-PCIe-80GB, B200-SXM-192GB, A10G,
L40S, L4, A100-PCIe-80GB, RTX-4060-Ti, RTX-4060, RTX-3060-Ti, RTX-3070,
MI210, V100-SXM-32GB. GB200 split into `GB200-Superchip` and
`B200-SXM-192GB` (the old single `GB200-NVL72` conflated superchip vs
per-GPU numbers). H100-NVL per-die numbers.

Legacy names (`H100-80GB`, `A100-80GB`, `A100-40GB`, `V100-32GB`,
`GB200-NVL72`) still resolve through an alias map so old saved plans
continue to look up correctly.

### Changed — API (breaking, pre-1.0)

- `HardwareSpec.hbm_bandwidth_gb_s` → **`memory_bandwidth_gb_s`**. The
  field is not HBM-specific (GDDR-based cards like L40S and consumer RTX
  had been stored under a misleading name).
- `HardwareSpec` gained `peak_tflops_by_precision`,
  `interconnect_latency_us`, and a `peak_tflops_for(precision)` lookup
  method.
- `ModelConfig` gained `ffn_type` (`"swiglu"` or `"standard"`) and
  `ffn_intermediate_size` for accurate parameter counts on real models.
- `RooflineAnalyzer.predict_decode_latency()` now **requires
  `sequence_length`**. Callers that used to ignore context length need
  to pass the current KV-cache depth.
- `RooflineAnalyzer.calculate_flops_per_token()` accepts an optional
  `sequence_length` (defaults to 0 for the bulk-matmul-only estimate).
- `CostAnalyzer.DEFAULT_PRICING` keyed on new canonical GPU names
  (legacy aliases resolved via `GPUDatabase.get()`).
- CLI `--goal` renamed to `--optimization-goal` (with `--goal` kept as
  an argparse alias so older scripts keep working).
- CLI `--rps`, `--input-length`, `--output-length` now reject
  non-positive values at parse time rather than silently failing deeper
  in the stack.

### Added

- `tests/unit/core/test_physics_regression.py` — 11 pinned regression
  tests anchoring the physics engine to primary-source numbers.
- `kv_planner/__init__.py` now re-exports the stable public API
  (`ModelConfig`, `HardwareSpec`, `DeploymentPlanner`, etc.) so
  `from kv_planner import ModelConfig` works.
- `pyproject.toml` adds `pythonpath = ["src"]` under `[tool.pytest.ini_options]`
  so `pytest` runs directly from a fresh checkout without `pip install -e .`.
- CHANGELOG (this file).

### Fixed — tooling / polish

- `pyproject.toml` `requires-python` and README now both say `>= 3.10`.
- `application/planner.py` type annotation `dict[str, any]` (was a call
  to the `any` builtin) is now `dict[str, Any]`.
- Four `validate_*.py` scripts and `example_complete.py` at repo root
  had `sys.path.insert(0, "/home/hesham-haroun/kv-planner/src")` as an
  absolute path; replaced with `Path(__file__).parent / "src"`.
- Removed the unused direct dependencies `scipy` and `plotly` from
  `[project.dependencies]`. Still available as extras under `[stats]`
  and `[plots]`.

## [0.1.0] — initial release

First public version. See git log for details.
