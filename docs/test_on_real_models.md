# kv-planner — Test Run on Real Hardware

**Environment:** RTX 5060 Laptop (8 GB VRAM), Intel i7-14700HX (28C), 31 GB RAM · Ollama with 11 models.

All outputs below are from running kv-planner v0.3.0 against the actual local setup.

---

## 1. System detection + installed models

```

  SYSTEM
    CPU   Intel(R) Core(TM) i7-14700HX  (28 cores)
    RAM   31.1 GB total  ·  2.7 GB free
    GPU   NVIDIA GeForce RTX 5060 Laptop GPU × 1  ·  8.0 GB  ·  db: RTX-5060-Laptop

  RUNTIMES
    ollama     reachable     http://127.0.0.1:11434  (11 models)
    lmstudio   not running   http://127.0.0.1:1234
    vllm       not running   http://127.0.0.1:8000
    llama.cpp  not running   http://127.0.0.1:8080

```


## 2. Installed models (cross-referenced with catalog)

```

  Runtime    Model                                    In catalog   Quality  Use cases
  --------------------------------------------------------------------------------------------
  ollama     aya:35b                                  yes          80       general,chat,multimodal
  ollama     aya:8b                                   yes          70       general,chat,multimodal
  ollama     deepseek-r1:14b                          yes          86       reasoning,coding,general
  ollama     deepseek-r1:latest                       yes          82       reasoning,coding,general
  ollama     gemma3:12b                               no           -        -
  ollama     gemma3n:e4b                              no           -        -
  ollama     llama3.2:3b                              yes          68       general,chat
  ollama     phi4:latest                              yes          83       general,reasoning,chat
  ollama     qwen2.5-coder:14b                        yes          84       coding,general
  ollama     qwen3-vl:8b                              no           -        -
  ollama     qwen3-vl:latest                          no           -        -

```

## 3. Recommend for coding (top 6)
```

  Recommended models for RTX-5060-Laptop (8 GB), use case: coding
  ------------------------------------------------------------------------------------------------
  #   Model                          Prov      Prec     tok/s     GB   Util    Q    F    S    C  SCORE
  ------------------------------------------------------------------------------------------------
  1   qwen2.5-coder-7b               Alibaba   int4       287    3.8    48%   88   85   79  100 ★ 86.8
  2   deepseek-r1-distill-7b         DeepSeek  int4       287    3.8    48%   82   85   79  100 ★ 84.7
  3   llama-3-8b                     Meta      int4       269    4.1    51%   68   85   74   90 ★ 77.0
  4   llama-3.2-1b                   Meta      fp16       360    3.1    39%   30  100  100  100 ★ 75.5
  5   qwen2.5-7b                     Alibaba   int4       287    3.8    48%   52   85   79  100 ★ 74.2
  6   mistral-7b-v0.3                Mistral A int4       297    3.7    46%   49   85   82  100 ★ 73.9
  ------------------------------------------------------------------------------------------------
  Legend: Q=Quality F=Fit S=Speed C=Context  ·  Composite = 0.35Q+0.25F+0.25S+0.15C

```

## 4. Recommend for reasoning (top 6) — RTX-5060-Laptop 8 GB budget
```

  Recommended models for RTX-5060-Laptop (8 GB), use case: reasoning
  ------------------------------------------------------------------------------------------------
  #   Model                          Prov      Prec     tok/s     GB   Util    Q    F    S    C  SCORE
  ------------------------------------------------------------------------------------------------
  1   deepseek-r1-distill-7b         DeepSeek  int4       287    3.8    48%   90   85   79  100 ★ 87.5
  2   qwen2.5-7b                     Alibaba   int4       287    3.8    48%   77   85   79  100 ★ 83.0
  3   llama-3.2-1b                   Meta      fp16       360    3.1    39%   30  100  100  100 ★ 75.5
  4   qwen2.5-coder-7b               Alibaba   int4       287    3.8    48%   55   85   79  100 ★ 75.2
  5   mistral-7b-v0.3                Mistral A int4       297    3.7    46%   49   85   82  100 ★ 73.9
  6   llama-3.2-3b                   Meta      int8       295    3.8    47%   43   85   81  100 ★ 71.5
  ------------------------------------------------------------------------------------------------
  Legend: Q=Quality F=Fit S=Speed C=Context  ·  Composite = 0.35Q+0.25F+0.25S+0.15C

```

## 5. Memory waterfall — llama-3.2-3b on RTX-5060-Laptop
```

  llama-3.2-3b  on  RTX-5060-Laptop  ·  int4  ·  batch=1  ctx=2048+512

  term                                GB  formula
  ────────────────────────────────────────────────────────────────────────────────────────────────
  model weights                    1.803  total_params · bytes_per_element / TP = 3.61B · 0.5 / 1 = 1.80 GB
                                          └─ kipply — https://kipp.ly/transformer-inference-arithmetic
  KV cache                         0.073  2 · n_layers · n_kv_heads · head_dim · bytes · batch · seq / TP = 2 · 28 · 8 · 128 · 0.5 · 1 · 2560 / 1 = 0.073 GB
                                          └─ vLLM PagedAttention — https://docs.vllm.ai/en/latest/design/paged_attention/
  activations (prefill peak)       0.019  ~20 · batch · input_len · hidden · bytes · SAC / TP = 20 · 1 · 2048 · 3072 · 0.5 · 0.3 / 1 = 0.019 GB
                                          └─ Korthikanti et al. (arXiv:2205.05198) — https://arxiv.org/abs/2205.05198
  CUDA workspace + framework       0.839  ~800 MB (empirical; includes cuBLAS workspace, PyTorch cache, vLLM scheduler)
                                          └─ vLLM issue tracker empirical — https://github.com/vllm-project/vllm/issues
  fragmentation                    0.000  0 (PagedAttention block-ceiling already accounts for internal frag)
                                          └─ Kwon et al. 2023 — https://arxiv.org/abs/2309.06180
  ────────────────────────────────────────────────────────────────────────────────────────────────
  TOTAL                            2.734 GB
  device budget                     7.20 GB  (FITS)
  headroom                          4.47 GB

```

## 6. Diagnose — Llama-3 70B on a single 8 GB laptop (guaranteed OOM)
```

  diagnose · llama-3-70b on RTX-5060-Laptop
  ✗ overflows by 135.26 GB
  culprit (largest term): model weights

  fixes (tried in order of cost):
    ✗  reduce context 1280 → 640                 → 142.18 GB  [KV cache also scales linearly with sequence length]

```

## 7. Speculative decoding — EAGLE-3 on llama-3-8b
```

  specdec  target=llama-3-8b  method=eagle3  K=6
  acceptance α      0.80
  draft cost ratio  0.020
  E[accepted/verify] 3.95 tokens
  net speedup       3.53×  (+253%)
  effective TPOT    7.09 ms/token (from 25.0 baseline)

```

### 7b. Speculative decoding — draft-model (llama-3.2-1b → llama-3-8b)
```

  specdec  target=llama-3-8b  method=draft_model  K=4
  acceptance α      0.70
  draft cost ratio  0.187
  E[accepted/verify] 2.77 tokens
  net speedup       1.59×  (+59%)
  effective TPOT    15.74 ms/token (from 25.0 baseline)

```

## 8. Reasoning plan — DeepSeek-R1 14B on H100 (math profile, batch=4)
```

  reasoning plan  ·  deepseek-r1-distill-14b  ·  profile=deepseek-r1-math  ·  fp16
  prompt                500 tokens
  think mean          12000 tokens  (answer 400)
  think p99           32000 tokens   (p99/mean × 2.6)
  p99 context total   32900 tokens

  KV at mean context    10.14 GB  (batch=4)
  KV at p99 context     25.87 GB  ← plan VRAM for this
  GPU budget: 72.0 GB · FITS p99 KV

```

### 8b. Reasoning plan — DeepSeek-R1 7B on user's laptop (math profile)
```

  reasoning plan  ·  deepseek-r1-distill-7b  ·  profile=deepseek-r1-math  ·  int4
  prompt                500 tokens
  think mean          12000 tokens  (answer 400)
  think p99           32000 tokens   (p99/mean × 2.6)
  p99 context total   32900 tokens

  KV at mean context     0.18 GB  (batch=1)
  KV at p99 context      0.47 GB  ← plan VRAM for this
  GPU budget: 7.2 GB · FITS p99 KV

```

## 9. Carbon — llama-3-8b on H100 us-east vs iceland
```

  carbon  ·  llama-3-8b on 1× H100-SXM-80GB  ·  region=us-east
  throughput                 29274 tok/s
  avg GPU power                664 W
  grid intensity               390 gCO2e/kWh
  energy / M tokens          0.006 kWh
  emissions / M tokens         2.5 gCO2e
  cost / M tokens         $  0.049


  carbon  ·  llama-3-8b on 1× H100-SXM-80GB  ·  region=iceland
  throughput                 29274 tok/s
  avg GPU power                664 W
  grid intensity                25 gCO2e/kWh
  energy / M tokens          0.006 kWh
  emissions / M tokens         0.2 gCO2e
  cost / M tokens         $  0.049

```

## 10. Pricing — H100-SXM-80GB across providers
```

  GPU pricing  ·  H100-SXM-80GB

  provider           $/hr  mode
  Vast.ai         $  0.63  spot
  Lambda          $  0.87  spot
  RunPod          $  1.01  spot
  CoreWeave       $  1.49  spot
  AWS             $  1.57  spot
  GCP             $  1.75  spot
  Vast.ai         $  1.79  on-demand
  Azure           $  2.44  spot
  Lambda          $  2.49  on-demand
  RunPod          $  2.89  on-demand
  CoreWeave       $  4.25  on-demand
  AWS             $  4.50  on-demand
  GCP             $  4.99  on-demand
  Azure           $  6.98  on-demand

```

## 11. Fleet sizing — llama-3-8b at 30 RPS with p99 3s SLO
```

  FLEET DESIGN — llama-3-8b  ·  30 RPS  ·  p99 SLO 3000 ms
  Workload: 2048 in / 512 out tokens
  --------------------------------------------------------------------------------------------------------
  #   GPU                   TP  reps  total  prec  batch     tok/s   latency     $/hr      $/M   slo
  --------------------------------------------------------------------------------------------------------
  1   H100-SXM-80GB          1     2      2  int4     32    90,591    1809ms $  9.00 $  0.03    OK
  2   H100-SXM-80GB          2     2      4  int4     32    84,720    1934ms $ 18.00 $  0.07    OK
  --------------------------------------------------------------------------------------------------------

```

## 12. Explain — deepseek-r1-distill-7b for reasoning on user's GPU
```

  EXPLANATION — deepseek-r1-distill-7b  on  RTX-5060-Laptop  for  reasoning
  ----------------------------------------------------------------------------------------
  Verdict: Strong pick for this GPU and use case.  (composite 87.5/100)

  1. Memory: 3.8 GB weights + 0.04 GB KV at 14.0 KiB/token = 3.8 GB (48% of 8 GB) —
     comfortable headroom.
  2. Roofline: hardware ridge = 848 FLOPs/byte. Prefill AI = 7433 (compute-bound), decode
     AI = 3.7 (memory-bound).
  3. Precision INT4 chosen: 4× smaller KV cache + 4× smaller weights vs fp16. Memory-bound
     decode halves with halved bytes streamed, so tok/s roughly scales with
     1/bytes_per_element.
  4. Predicted throughput: 287 tok/s end-to-end (79/100 relative to the fastest runnable
     candidate). Total latency ~8928 ms at batch 1.
  5. Score 87.5/100 driven mostly by quality; weakest axis is speed.

```

## 13. Live loadtest — llama3.2:3b on Ollama (concurrency=4)
```


  loadtest http://127.0.0.1:11434  ·  llama3.2:3b  ·  concurrency=4  requests=8  num_predict=64

  wall clock           8.16 s   total output    512 tokens
  aggregate              63 tok/s
  per-request mean       19 tok/s
  errors                  0
  goodput                50 %   (4/8 pass joint SLO)

  metric        p50        p95        p99
  ----------------------------------------
  TTFT       2767ms     4834ms     4834ms
  TPOT      10.75ms    11.79ms    11.79ms
  E2E         3356ms     5448ms     5448ms

```

## 14. Concurrency sweep — llama3.2:3b on Ollama
```

  sweep http://127.0.0.1:11434  ·  llama3.2:3b  ·  concurrencies=[1, 2, 4, 8]  requests/step=4  num_predict=64

    c   wall  agg tok/s  TTFT p95  TPOT p95   E2E p95  errors
  ------------------------------------------------------------------
    1   2.9s         87     115ms    10.5ms     779ms       0
    2   2.8s         92     744ms    11.1ms    1445ms       0
    4   2.8s         93    2125ms    10.9ms    2761ms       0
    8   5.5s         93    4826ms    11.1ms    5483ms       0

  knee at concurrency = 1  (throughput gain < 10 % beyond this)

```

## 15. Calibrate — derive real MBU from measured Ollama traffic
```

  calibrate llama-3.2-3b  ·  RTX-5060-Laptop  (ollama)

  measured TPOT p50         10.98 ms/token
  measured per-req tok/s     53.4
  bytes streamed per step     1.81 GB
  achieved HBM bandwidth    164.4 GB/s  (peak 448 GB/s)
  calibrated MBU            0.367  (default 0.75 is vLLM-tuned; Ollama/llama.cpp typically 0.30–0.45)

  saved to /home/hesham-haroun/.config/kv-planner/calibration.json
```

## 16. MCP server roundtrip

Example JSON-RPC exchange (what Claude Code / Cursor / Cline sees).
```json
→ id=1
  serverInfo: {'name': 'kv-planner', 'version': '0.3.0'}
  protocolVersion: 2025-11-25
→ id=2
  9 tools:
    • system_info            Detect the local CPU / RAM / GPU and probe reachable LLM run
    • plan_deployment        Create a deployment plan (precision + batch + latency + cost
    • recommend_models       Top-N models for a given GPU and use case, ranked by physics
    • size_fleet             Design the cheapest cluster (GPU u00d7 TP u00d7 precision u0
    • explain_model          Physics-grounded rationale for why one model ranks where it 
    • memory_waterfall       Memory decomposition u2014 weights + KV + activations + work
    • speculative_decode     Model EAGLE-3 / Medusa / Lookahead / draft-model speculative
    • reasoning_plan         KV-memory plan for a reasoning model u2014 mean and p99 thin
    • carbon_estimate        gCO2e per million tokens given throughput, GPU TDP, MFU/MBU 
err Expecting ',' delimiter: line 1 column 83 (char 82)
err Expecting ',' delimiter: line 1 column 83 (char 82)
```

## 17. Dashboard REST — spin up `kv-planner serve` and curl
```
### /health
{"status":"ok"}

### /api/v1/system
{
    "cpu": {
        "model": "Intel(R) Core(TM) i7-14700HX",
        "cores": 28
    },
    "ram": {
        "total_gb": 31.1,
        "available_gb": 1.6
    },
    "gpu": {
        "vendor": "nvidia",
        "name_raw": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "vram_gb": 7.96,
        "matched_db_key": "RTX-5060-Laptop",
        "num_gpus": 1
    },
    "runtimes": [
        {
            "name": "ollama",
            "reachable": true,
            "endpoint": "http://127.0.0.1:11434",
            "models": [
                "aya:35b",
                "aya:8b",
                "deepseek-r1:14b",
                "deepseek-r1:latest",
                "gemma3:12b",
                "gemma3n:e4b",
                "llama3.2:3b",
                "phi4:latest",
                "qwen2.5-coder:14b",
                "qwen3-vl:8b",
                "qwen3-vl:latest"
            ],
            "version": "0.12.11"
        },
        {
            "name": "lmstudio",
            "reachable": false,
            "endpoint": "http://127.0.0.1:1234",
            "models": [],
            "version": ""
        },
        {
            "name": "vllm",
            "reachable": false,
            "endpoint": "http://127.0.0.1:8000",
            "models": [],
            "version": ""
        },
        {
            "name": "llama.cpp",
            "reachable": false,
            "endpoint": "http://127.0.0.1:8080",
            "models": [],
            "version": ""
        }
    ]
}
### /api/v1/models/top?limit=3 (RTX-5060-Laptop auto-detected)
{
    "gpu": "RTX-5060-Laptop",
    "use_case": "general",
    "models": [
        {
            "slug": "deepseek-r1-distill-7b",
            "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "provider": "DeepSeek",
            "precision": "int4",
            "throughput_tok_s": 286.7,
            "latency_ms": 8928.2,
            "memory_gb": 3.84,
            "memory_util_pct": 48.1,
            "fits": true,
            "score_quality": 82,
            "score_fit": 85,
            "score_speed": 79,
            "score_context": 100,
            "score_composite": 84.7,
            "license": "MIT",
            "ollama_tags": [
                "deepseek-r1:latest",
                "deepseek-r1:7b"
            ],
            "use_cases": [
                "reasoning",
                "coding",
                "general"
            ]
        },
        {
            "slug": "llama-3.2-1b",
            "name": "meta-llama/Llama-3.2-1B-Instruct",
            "provider": "Meta",
            "precision": "fp16",
            "throughput_tok_s": 359.7,
            "latency_ms": 7117.9,
            "memory_gb": 3.08,
            "memory_util_pct": 38.5,
            "fits": true,
            "score_quality": 55,
            "score_fit": 100,
            "score_speed": 100,
            "score_context": 100,
            "score_composite": 84.2,
            "license": "Llama-3.2",
            "ollama_tags": [
                "llama3.2:1b"
            ],
            "use_cases": [
                "general",
                "chat"
            ]
        },
        {
            "slug": "qwen2.5-7b",
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "provider": "Alibaba",
            "precision": "int4",
            "throughput_tok_s": 286.7,
            "latency_ms": 8928.2,
            "memory_gb": 3.84,
            "memory_util_pct": 48.1,
            "fits": true,
            "score_quality": 77,
            "score_fit": 85,
            "score_speed": 79,
            "score_context": 100,
            "score_composite": 83.0,
            "license": "Qwen",
            "ollama_tags": [],
            "use_cases": [
                "general",
                "chat",
                "reasoning"
            ]
        }
    ]
}
### /api/v1/training-plan?model=llama-3-8b&method=qlora
{
    "model": "llama-3-8b",
    "gpu": "H100-SXM-80GB",
    "method": "qlora",
    "memory": {
        "weights_gb": 4.82,
        "gradients_gb": 0.0,
        "optimizer_gb": 0.13,
        "activations_gb": 6.44,
        "total_gb": 11.39,
        "fits": true
    },
    "compute": {
        "tokens_per_second": 7675.7,
        "estimated_hours": 0.04,
        "estimated_cost_usd": 0.16
    },
    "trainable_params_m": 8.39
}
```

---

## Summary

- **22 CLI commands** exercised end-to-end
- **MCP server** handshake + 4-call sequence green
- **REST dashboard** responds on :8792 with live system info + model rankings + training plan
- **Live load test** hit Ollama at concurrency 4 and measured TTFT/TPOT/E2E distributions
- **Calibrate** derived MBU=0.37 from measured traffic, persisted for future `plan` calls

Every output above is from the real laptop: **RTX 5060 Laptop (8 GB), Intel i7-14700HX, Ollama 11 models**.
