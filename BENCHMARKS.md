# kv-planner — Benchmark Record

This document is the **honest validation ledger** for kv-planner's
predictions. Every row distinguishes *measured on real hardware* from
*published-number reference* from *theoretical roofline output*.

It exists because previous iterations of this README claimed "90–95 %
accuracy" based on one GPU. That's not validation — it's curve-fitting
on a single data point. This page is what validation would actually
look like. It is deliberately short right now.

---

## Validation coverage at a glance

| Category | Count | Comment |
|---|---|---|
| GPUs with real end-to-end measurement | **1** (RTX 5060 Laptop) | via Ollama local runtime |
| GPUs with published-benchmark comparison | 0 (planned) | vLLM / SGLang / llm-d numbers pending |
| GPUs with only theoretical roofline | 36 | everything in `gpu_specs.py` except RTX 5060 Laptop |
| Dense models measured | 3 (Llama-3.2-3B, DeepSeek-R1-Distill-7B, Aya-8B) | |
| MoE models measured | 0 | Mixtral, Qwen3-MoE, DeepSeek-V2 — theory only |
| Total predictive samples | ~3 | |

**One GPU is not a validation set.** The rest of this file enumerates
what we do have, what's next, and how to reproduce each number.

---

## 1. Real-hardware measurements (Ollama on RTX 5060 Laptop)

Host: Intel i7-14700HX · 31 GB RAM · RTX 5060 Laptop 8 GB · Ollama 0.12.11 ·
Linux 6.17.

Full captured output: [`docs/test_on_real_models.md`](docs/test_on_real_models.md).

Benchmark script: [`benchmark_ollama.py`](benchmark_ollama.py) — reproducible
with `python3 benchmark_ollama.py`.

### Dense decode throughput

| Model | Quant | Measured (tok/s) | Predicted @ default MBU=0.75 (tok/s) | Predicted @ MBU=0.35 (calibrated for llama.cpp) (tok/s) | MAPE (calibrated) |
|---|---|---:|---:|---:|---:|
| Llama-3.2 3B | Q4_K_M | **98.4** | 185.7 | 88.8 | 9.8 % |
| DeepSeek-R1-Distill 7B | Q4_K_M | **45.4** | 88.2 | 39.4 | 13.2 % |
| Aya-8B (Command-R) | Q4_K_M | **53.3** | 73.9 | ~38 | ~29 % |

**Notes:**

- Measurements from Ollama's own nanosecond timings (`eval_duration`,
  `prompt_eval_duration`) — not HTTP wall clock.
- Default `memory_efficiency=0.75` is tuned for vLLM+FlashAttention
  kernels. For llama.cpp / Ollama on a laptop GPU, the realized MBU
  measured here is **0.37** — derived by `kv-planner calibrate`.
- After calibration to 0.37, two of three models predict within 15 %.
- **Aya-8B has 29 % error** because Command-R uses tied embeddings that
  `total_params()` doesn't account for. Open issue. Not hidden.

### Concurrency sweep (Llama-3.2 3B)

Prediction matches reality's *shape*, not just point values — the
saturation knee is at concurrency 1:

| Concurrency | Measured agg tok/s | Measured p95 TTFT | Predicted saturation |
|---:|---:|---:|---|
| 1 | 87 | 115 ms | knee detected here |
| 2 | 92 | 739 ms | < 10 % gain (correct) |
| 4 | 93 | 2 125 ms | < 1 % gain (correct) |
| 8 | 93 | 4 826 ms | < 1 % gain (correct) |

The sweep correctly identifies that Ollama-on-laptop is single-request
bound. Predictions for laptop + Ollama should assume concurrency 1, not
aggregate capacity.

### Calibration-loop validation

`kv-planner calibrate` back-solves MBU from measured TPOT. The derived
value 0.37 matches the published literature range for llama.cpp on
laptop GPUs (0.30–0.45 per Red Hat's vLLM vs llama.cpp blogs, BentoML
benchmarks, SGLang vs llama.cpp comparisons). That's a sanity check —
not independent validation.

---

## 2. Physics regression fixtures

These anchor individual formulas to published reference points. They
are not end-to-end validation — they are "does the equation output the
number the paper says it does" checks.

Run with `pytest tests/unit/core/test_physics_regression.py tests/unit/core/test_moe.py`.

| Fixture | What it pins | Reference |
|---|---|---|
| `test_llama3_8b_kv_cache_bytes_per_token_fp16` | 131 072 bytes (128 KiB) | LMCache KV calculator; HF config |
| `test_llama3_8b_parameter_count_within_5pct` | 8.03 B ± 5 % | Meta Llama-3 card |
| `test_llama3_70b_parameter_count_within_5pct` | 70.6 B ± 5 % | Meta Llama-3 card |
| `test_h100_balance_point` | 295 FLOPs/byte | jax-ml scaling book |
| `test_llama3_8b_h100_throughput_sanity` | 500–40 000 tok/s on H100 | wide range; just a sanity bound |
| `test_paged_fragmentation_exact_at_partial_block` | 46.9 % at seq_len=17 | vLLM paper + 1/block_size math |
| `test_paged_fragmentation_zero_at_exact_multiple` | 0 % when seq_len = 16·k | ceiling-block accounting |
| `test_decode_latency_scales_with_kv_cache` | long ctx > 1.5× short ctx | the 0.2.0 bug-fix regression |
| `test_tp_allreduce_sublinear` | TP=8 cost < 4× TP=2 | ring AllReduce asymptotics |
| `test_super_linear_scaling_method_removed` | no 13.9× OOM bug | regression guard |
| `test_mixtral_active_is_about_13b` | 11.5–14 B active | Mistral publishes 12.9 B |
| `test_mixtral_total_is_about_47b` | 45–50 B total | Mistral publishes 46.7 B |
| `test_qwen3_30b_a3b_ratio_matches_name` | 30 B / ~3 B active | "A3B" = 3 B active per name |
| `test_mixtral_decode_faster_than_40b_dense` | Mixtral decode < Llama-70B decode | consequence of active vs total |

141 tests run in ~0.5 s.

---

## 3. GPUs with zero measurement

These are theoretical roofline output only. Use at your own risk for
sizing:

H100-SXM-80GB · H100-PCIe-80GB · H100-NVL-94GB · H200-SXM-141GB ·
B200-SXM-192GB · GB200-Superchip · A100-SXM-40GB · A100-SXM-80GB ·
A100-PCIe-80GB · A10G · L40S · L4 · V100-SXM-32GB · RTX-5090 ·
RTX-5080 · RTX-5070-Ti · RTX-5070 · RTX-4090 · RTX-4080-Super ·
RTX-4080 · RTX-4070-Ti-Super · RTX-4070-Ti · RTX-4070-Super · RTX-4070 ·
RTX-4060-Ti · RTX-4060 · RTX-3090-Ti · RTX-3090 · RTX-3080-Ti ·
RTX-3080-12GB · RTX-3080 · RTX-3070-Ti · RTX-3070 · RTX-3060-Ti ·
MI300X · MI250X · MI210.

kv-planner's prediction for any of these is the roofline formula's
output. Real tok/s may be 0.5× to 1.0× of prediction depending on
kernel quality (vLLM/SGLang close the gap; llama.cpp / naive
implementations do not).

---

## 4. How this page grows

### Pending: RunPod validation campaign (harness ready, awaiting run)

The [`scripts/validation/`](scripts/validation/) harness can run
kv-planner predictions against live vLLM servers on rented GPUs and
write structured result JSONs for every (GPU × model × precision)
configuration.

Default matrix (~$3.50 estimated RunPod spend):

| GPU | Model | Precision | Est. min | Est. $ |
|---|---|---|---:|---:|
| H100-SXM-80GB | Llama-3-8B | fp16 | 18 | $1.20 |
| H100-SXM-80GB | Qwen2.5-7B | fp16 | 18 | $1.20 |
| A100-PCIe-80GB | Llama-3-8B | fp16 | 18 | $0.57 |
| L40S | Llama-3-8B | fp16 | 18 | $0.36 |
| RTX-4090 | Llama-3-8B | fp16 | 18 | $0.13 |

Extended MoE matrix adds Mixtral-8x7B on H100 (~$2 more).

Reproduction (two paths documented in
[`scripts/validation/README.md`](scripts/validation/README.md)):

```bash
# Path A — fully automated with RunPod API
export RUNPOD_API_KEY=... HF_TOKEN=...
python scripts/validation/runpod_orchestrator.py --budget-usd 25
python scripts/validation/aggregate_results.py

# Path B — manual one-pod-at-a-time
# (SSH into a RunPod pod, run in_pod_validate.sh, scp result back)
```

Each pod runs `scripts/validation/in_pod_validate.sh` which installs
vLLM, serves the target model, hits it with `kv-planner loadtest`,
back-solves MBU with `kv-planner calibrate`, and emits one JSON
containing: config, predicted numbers, measured numbers,
calibration-derived MBU, and MAPE (TPOT + throughput).

**As of this commit the campaign has not been run.** When it does, this
section gets replaced by the measured table.

### Roadmap beyond this campaign

| Milestone | What gets added |
|---|---|
| **0.4.0** | MoE campaign (Mixtral / Qwen3-30B-A3B / DeepSeek-V2-Lite); chunked-prefill modeling; SGLang runtime support alongside vLLM. |
| **0.5.0** | Disaggregated prefill/decode planner; mutation-testing audit of the roofline test suite. |

Until the RunPod campaign lands with real H100 / A100 / L40S MAPE
numbers, kv-planner remains a **research prototype** — well-cited
physics engine, narrow empirical coverage.

---

## Contributing a measurement

If you run kv-planner predictions against a real deployment on any
GPU in the list above, please submit a PR to this file with:

1. Host specs (GPU, OS, runtime, runtime version)
2. Model + quantization
3. Measured TPOT / p50 throughput / MBU (calibrate command output)
4. Predicted TPOT at default MBU=0.75
5. MAPE after calibration

Even one row on H100 would materially improve the validation coverage
this file records.
