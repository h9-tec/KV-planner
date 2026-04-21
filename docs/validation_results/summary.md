# kv-planner — RunPod validation campaign results

Generated: 2026-04-21T10:05:10Z
Configs measured: **4**

## Summary table

| GPU | Model | Precision | Predicted tok/s | Measured tok/s | Predicted TPOT ms | Measured TPOT ms | MAPE TPOT | MBU derived |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A100-PCIe-80GB | qwen2.5-7b | fp16 | 797 | 703 | 10.6 | 10.6 | 0.6% | None |
| H100-SXM-80GB | deepseek-v2-lite | fp16 | 4350 | 1225 | 2.0 | 5.0 | 60.0% | None |
| H100-SXM-80GB | qwen2.5-7b | fp16 | 1420 | 1093 | 6.1 | 6.0 | 1.9% | None |
| RTX-4090 | qwen2.5-7b | fp16 | 416 | 474 | 20.3 | 15.8 | 28.1% | None |

## Per-config detail

### A100-PCIe-80GB — qwen2.5-7b (fp16)

```json
{
  "config": {
    "gpu_key": "A100-PCIe-80GB",
    "model_hf": "Qwen/Qwen2.5-7B-Instruct",
    "model_slug": "qwen2.5-7b",
    "precision": "fp16",
    "input_length": 2048,
    "output_length": 256,
    "concurrency": 8,
    "num_requests": 32,
    "runtime": "vllm"
  },
  "predicted": {
    "throughput_tok_s": 797.010920411384,
    "prefill_ms": 182.12313245538462,
    "decode_total_ms": 2708.677910535056,
    "tpot_ms": 10.580773088027563,
    "mfu": 0.5,
    "mbu": 0.75,
    "ai": 1858.2429188670187,
    "compute_bound_prefill": true
  },
  "measured": {
    "aggregate_tok_s": 703.3,
    "ttft_ms_p50": 253.0,
    "ttft_ms_p99": 442.9,
    "tpot_ms_p50": 10.64,
    "tpot_ms_p99": 10.64,
    "e2e_ms_p50": 2966.3,
    "e2e_ms_p99": 3153.3,
    "errors": 0,
    "raw_keys": [
      "aggregate_tok_s",
      "api",
      "concurrency",
      "e2e_ms",
      "endpoint",
      "errors",
      "goodput_pct",
      "model",
      "num_requests",
      "per_request_tok_s_mean",
      "total_output_tokens",
      "tpot_ms",
      "ttft_ms",
      "wall_s"
    ]
  },
  "calibration": {
    "derived_mbu": null,
    "achieved_bandwidth_gb_s": null,
    "peak_bandwidth_gb_s": null
  },
  "accuracy": {
    "mape_tpot_pct": 0.6,
    "mape_throughput_pct": 13.3
  }
}
```

### H100-SXM-80GB — deepseek-v2-lite (fp16)

```json
{
  "config": {
    "gpu_key": "H100-SXM-80GB",
    "model_hf": "deepseek-ai/DeepSeek-V2-Lite",
    "model_slug": "deepseek-v2-lite",
    "precision": "fp16",
    "input_length": 2048,
    "output_length": 256,
    "concurrency": 8,
    "num_requests": 32,
    "runtime": "vllm"
  },
  "predicted": {
    "throughput_tok_s": 4350.117378787113,
    "prefill_ms": 17.236324770297273,
    "decode_total_ms": 512.4045560103483,
    "tpot_ms": 2.001580296915423,
    "mfu": 0.49999999999999994,
    "mbu": 1.0,
    "ai": 269.08474576271186,
    "compute_bound_prefill": false
  },
  "measured": {
    "aggregate_tok_s": 1225.3,
    "ttft_ms_p50": 67.9,
    "ttft_ms_p99": 1228.5,
    "tpot_ms_p50": 5.01,
    "tpot_ms_p99": 5.62,
    "e2e_ms_p50": 1498.1,
    "e2e_ms_p99": 2496.9,
    "errors": 0,
    "raw_keys": [
      "aggregate_tok_s",
      "api",
      "concurrency",
      "e2e_ms",
      "endpoint",
      "errors",
      "goodput_pct",
      "model",
      "num_requests",
      "per_request_tok_s_mean",
      "total_output_tokens",
      "tpot_ms",
      "ttft_ms",
      "wall_s"
    ]
  },
  "calibration": {
    "derived_mbu": null,
    "achieved_bandwidth_gb_s": null,
    "peak_bandwidth_gb_s": null
  },
  "accuracy": {
    "mape_tpot_pct": 60.0,
    "mape_throughput_pct": 255.0
  }
}
```

### H100-SXM-80GB — qwen2.5-7b (fp16)

```json
{
  "config": {
    "gpu_key": "H100-SXM-80GB",
    "model_hf": "Qwen/Qwen2.5-7B-Instruct",
    "model_slug": "qwen2.5-7b",
    "precision": "fp16",
    "input_length": 2048,
    "output_length": 256,
    "concurrency": 8,
    "num_requests": 32,
    "runtime": "vllm"
  },
  "predicted": {
    "throughput_tok_s": 1420.4518136524061,
    "prefill_ms": 57.4544159009909,
    "decode_total_ms": 1564.5647035478607,
    "tpot_ms": 6.111580873233831,
    "mfu": 0.5,
    "mbu": 0.75,
    "ai": 1858.2429188670187,
    "compute_bound_prefill": true
  },
  "measured": {
    "aggregate_tok_s": 1093.3,
    "ttft_ms_p50": 51.6,
    "ttft_ms_p99": 796.5,
    "tpot_ms_p50": 6.0,
    "tpot_ms_p99": 7.67,
    "e2e_ms_p50": 1967.9,
    "e2e_ms_p99": 2329.8,
    "errors": 0,
    "raw_keys": [
      "aggregate_tok_s",
      "api",
      "concurrency",
      "e2e_ms",
      "endpoint",
      "errors",
      "goodput_pct",
      "model",
      "num_requests",
      "per_request_tok_s_mean",
      "total_output_tokens",
      "tpot_ms",
      "ttft_ms",
      "wall_s"
    ]
  },
  "calibration": {
    "derived_mbu": null,
    "achieved_bandwidth_gb_s": null,
    "peak_bandwidth_gb_s": null
  },
  "accuracy": {
    "mape_tpot_pct": 1.9,
    "mape_throughput_pct": 29.9
  }
}
```

### RTX-4090 — qwen2.5-7b (fp16)

```json
{
  "config": {
    "gpu_key": "RTX-4090",
    "model_hf": "Qwen/Qwen2.5-7B-Instruct",
    "model_slug": "qwen2.5-7b",
    "precision": "fp16",
    "input_length": 2048,
    "output_length": 256,
    "concurrency": 8,
    "num_requests": 32,
    "runtime": "vllm"
  },
  "predicted": {
    "throughput_tok_s": 415.61023625354596,
    "prefill_ms": 343.96136395932206,
    "decode_total_ms": 5199.69420325926,
    "tpot_ms": 20.311305481481483,
    "mfu": 0.5,
    "mbu": 0.75,
    "ai": 1858.2429188670187,
    "compute_bound_prefill": true
  },
  "measured": {
    "aggregate_tok_s": 474.2,
    "ttft_ms_p50": 365.5,
    "ttft_ms_p99": 603.8,
    "tpot_ms_p50": 15.85,
    "tpot_ms_p99": 15.87,
    "e2e_ms_p50": 4409.2,
    "e2e_ms_p99": 4644.8,
    "errors": 0,
    "raw_keys": [
      "aggregate_tok_s",
      "api",
      "concurrency",
      "e2e_ms",
      "endpoint",
      "errors",
      "goodput_pct",
      "model",
      "num_requests",
      "per_request_tok_s_mean",
      "total_output_tokens",
      "tpot_ms",
      "ttft_ms",
      "wall_s"
    ]
  },
  "calibration": {
    "derived_mbu": null,
    "achieved_bandwidth_gb_s": null,
    "peak_bandwidth_gb_s": null
  },
  "accuracy": {
    "mape_tpot_pct": 28.1,
    "mape_throughput_pct": 12.4
  }
}
```
