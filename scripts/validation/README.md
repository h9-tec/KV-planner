# kv-planner RunPod validation harness

End-to-end validation on rented GPUs. Two paths: fully automated (if you
have a RunPod API key and SSH key) or manual one-pod-at-a-time.

## What this measures

For each (GPU, model, precision) configuration:

1. Start vLLM as an OpenAI-compatible server.
2. Ask kv-planner what it predicts for that workload.
3. Run `kv-planner loadtest` against the live vLLM and capture real
   TTFT / TPOT / E2E / aggregate tok/s.
4. Run `kv-planner calibrate` to derive the runtime's actual MBU.
5. Emit one JSON with `(predicted, measured, mape_tpot_pct, derived_mbu)`.

Result artifacts aggregate into `docs/validation_results/summary.md`
which replaces the current "1 GPU only" status in `BENCHMARKS.md`.

---

## Path A — fully automated

Requires RunPod account + API key and an SSH key added in the RunPod
console.

```bash
export RUNPOD_API_KEY=rpa_...
export HF_TOKEN=hf_...            # for gated models (Llama family)

# First run: print the matrix and estimated cost, then exit
python scripts/validation/runpod_orchestrator.py

# Actually launch (give it a hard budget cap):
python scripts/validation/runpod_orchestrator.py --budget-usd 25

# Include MoE (heavier; Mixtral-8x7B on H100)
python scripts/validation/runpod_orchestrator.py --budget-usd 35 --include-moe
```

Default matrix is 5 configs:

| GPU | Model | Est. min | Est. $ |
|---|---|---:|---:|
| H100-SXM-80GB | llama-3-8b | 18 | $1.20 |
| H100-SXM-80GB | qwen2.5-7b | 18 | $1.20 |
| A100-PCIe-80GB | llama-3-8b | 18 | $0.57 |
| L40S | llama-3-8b | 18 | $0.36 |
| RTX-4090 | llama-3-8b | 18 | $0.13 |
| **Total** | | **~90** | **~$3.46** |

The orchestrator hard-terminates every pod it creates — even on script
crash — and caps total spend at `--budget-usd`.

### Results

Each configuration drops a JSON into `docs/validation_results/`. Merge
them into one markdown table:

```bash
python scripts/validation/aggregate_results.py \
    --results-dir docs/validation_results \
    --output docs/validation_results/summary.md
```

Copy the interesting rows into `BENCHMARKS.md` → "Real-hardware
measurements (enterprise GPUs)".

---

## Path B — manual one-pod-at-a-time

If you prefer to provision pods in the RunPod UI:

1. In the console, create an on-demand pod with the target GPU +
   template `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
   expose ports `8000/http` and `22/tcp`, 30 GB container disk, 60 GB
   volume on `/workspace`.
2. SSH into the pod and run:

```bash
cd /workspace
git clone https://github.com/h9-tec/KV-planner.git repo && cd repo
export HF_TOKEN=hf_...

# Pick one:
bash scripts/validation/in_pod_validate.sh \
    meta-llama/Meta-Llama-3-8B-Instruct  H100-SXM-80GB  llama-3-8b  fp16
```

The script takes ~15-20 min (install + download + warmup + testing).
The final line is the path to `validation_result.json` — `scp` it back
and drop it in `docs/validation_results/`, then run the aggregator.

3. Terminate the pod from the UI.

---

## Cost controls

- Hard budget cap via `--budget-usd`; orchestrator stops before exceeding.
- Community-cloud pricing picked (cheaper than secure-cloud).
- 18-min-per-config estimate is conservative; typically ~12 min.
- If any pod doesn't reach RUNNING in 10 min, it's auto-terminated.
- Every pod has a guaranteed-terminate `finally` clause — even on
  orchestrator crash.

## Safety

- Read-only on the repo tree; writes only to `docs/validation_results/`.
- HF_TOKEN is passed through env, never logged.
- No pod opens inbound ports other than SSH (22) + vLLM :8000.
- vLLM is on 0.0.0.0 inside the pod but RunPod's network is NAT-ed;
  only SSH is reachable from outside.

## Extending the matrix

Edit `DEFAULT_MATRIX` or `MOE_MATRIX` in `runpod_orchestrator.py`.
Each `Config` row is one pod; `estimated_pod_minutes` and
`estimated_hourly_usd` drive the budget math.

Adding a new model to test requires:

- A catalog entry in `src/kv_planner/infrastructure/model_catalog.py`
- The HuggingFace model id (gated models need `HF_TOKEN` with access)
- A GPU entry in `src/kv_planner/infrastructure/hardware_db/gpu_specs.py`
- A RunPod GPU type ID (see `runpod.io/console/gpus` or `list_available_gpus()`)
