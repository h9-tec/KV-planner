#!/usr/bin/env bash
# in_pod_validate.sh — runs inside a RunPod / Lambda / Vast.ai pod after SSH.
#
# What it does:
#   1. Ensures Python, vLLM, and kv-planner are installed.
#   2. Starts vLLM as an OpenAI-compatible server on :8000 with the target model.
#   3. Waits for the server to become healthy.
#   4. Runs kv-planner predictions for the same (model, gpu, workload).
#   5. Runs kv-planner loadtest + calibrate against the live vLLM.
#   6. Emits one JSON file that ties (predicted, measured, MBU_derived, MAPE) together.
#
# Usage (inside the pod):
#   ./in_pod_validate.sh  MODEL_HF  GPU_KEY  MODEL_SLUG  [PRECISION]
#
# Example:
#   ./in_pod_validate.sh meta-llama/Meta-Llama-3-8B-Instruct H100-SXM-80GB llama-3-8b fp16
#
# Output:
#   /workspace/validation_result.json   (single artifact to scp back)

set -euo pipefail

MODEL_HF="${1:?model HF id required (e.g. meta-llama/Meta-Llama-3-8B-Instruct)}"
GPU_KEY="${2:?GPU DB key required (e.g. H100-SXM-80GB)}"
MODEL_SLUG="${3:?kv-planner catalog slug required (e.g. llama-3-8b)}"
PRECISION="${4:-fp16}"

# Tunables
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
VLLM_PORT="${VLLM_PORT:-8000}"

cd "$(dirname "$0")"
REPO_DIR="$(cd .. && cd .. && pwd)"
WORK_DIR="${WORKSPACE:-/workspace}"
mkdir -p "$WORK_DIR"

log() { echo "[$(date +%T)] $*"; }

# ------- Step 1: Install -----------------------------------------------------
log "Step 1/6: ensuring Python + deps"
which python3 >/dev/null || { echo "python3 missing"; exit 1; }
python3 -c "import torch; print('base torch:', torch.__version__, 'CUDA:', torch.version.cuda)" || true
python3 -m pip install --quiet --upgrade pip

# Upgrade torch alongside vllm so we don't end up with torch 2.4 + vllm 0.9
# (vllm requires torch 2.5+). Pinning vllm to a version that's known-good on
# the current CUDA/driver combo is safer than letting pip pick "latest".
log "Step 1/6: installing vllm (this takes 5-8 min)"
python3 -m pip install --quiet --upgrade vllm
python3 -c "import vllm, torch; print('installed vllm:', vllm.__version__, '/ torch:', torch.__version__, '/ cuda avail:', torch.cuda.is_available())"

# Prove HF_TOKEN propagation without leaking the value.
if [ -n "${HF_TOKEN:-}" ]; then
    log "HF_TOKEN is set (length=${#HF_TOKEN})"
else
    log "WARNING: HF_TOKEN is NOT set — gated models will 401"
fi

# Put HuggingFace cache on the /workspace volume (60 GB) rather than the
# container disk (30 GB, partially consumed by pip installs). This is the
# specific fix for "No space left on device" seen on smoke v5.
export HF_HOME="$WORK_DIR/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
log "HF_HOME=$HF_HOME"
df -h "$WORK_DIR" /root 2>/dev/null || true

# kv-planner itself (already checked out before calling this script)
if [ -d "$REPO_DIR/src/kv_planner" ]; then
    python3 -m pip install --quiet -e "$REPO_DIR[tui]"
else
    python3 -m pip install --quiet 'kv-planner[tui]' || {
        echo "kv-planner not found — clone it first"
        exit 1
    }
fi

# ------- Step 2: Start vLLM --------------------------------------------------
log "Step 2/6: starting vLLM on :$VLLM_PORT with $MODEL_HF"
VLLM_LOG="$WORK_DIR/vllm.log"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_HF" \
    --dtype "$([ "$PRECISION" = "fp16" ] && echo float16 || echo auto)" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# ------- Step 3: Wait for ready ---------------------------------------------
log "Step 3/6: waiting for vLLM to come up (max 15 min)"
for i in $(seq 1 90); do
    sleep 10
    if curl -s -f "http://127.0.0.1:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        log "vLLM is up after ${i}0s"
        break
    fi
    # Also fail fast if vLLM already crashed — no point waiting 15 min.
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vLLM subprocess exited early. Full log below:"
        echo "================= vllm.log (full) ================="
        cat "$VLLM_LOG" || true
        echo "================= end vllm.log ==================="
        exit 1
    fi
    if [ "$i" = "90" ]; then
        log "vLLM failed to start within 15 min. Full log below:"
        echo "================= vllm.log (full) ================="
        cat "$VLLM_LOG" || true
        echo "================= end vllm.log ==================="
        exit 1
    fi
done

# ------- Step 4: Get kv-planner predictions ---------------------------------
log "Step 4/6: computing kv-planner predictions"
PREDICTION_JSON="$WORK_DIR/prediction.json"
python3 -m kv_planner.cli.main plan \
    --model "$MODEL_HF" \
    --gpu "$GPU_KEY" \
    --rps 10 \
    --input-length "$INPUT_LEN" \
    --output-length "$OUTPUT_LEN" \
    --optimization-goal balanced \
    --format json > "$PREDICTION_JSON" 2>&1 || true

# Also grab the raw roofline prediction for one-sequence decode
python3 - <<EOF > "$WORK_DIR/roofline_raw.json"
import json
from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.model_catalog import by_slug

entry = by_slug("$MODEL_SLUG")
gpu = GPUDatabase.to_hardware_spec("$GPU_KEY")
ra = RooflineAnalyzer()
m = ra.predict_latency(
    model=entry.config, hardware=gpu, batch_size=1,
    input_length=$INPUT_LEN, output_length=$OUTPUT_LEN, precision="$PRECISION",
)
print(json.dumps({
    "throughput_tok_s": m.throughput_tokens_per_sec,
    "prefill_ms": m.prefill_latency_ms,
    "decode_total_ms": m.decode_latency_ms,
    "tpot_ms": m.decode_latency_ms / max(1, $OUTPUT_LEN),
    "mfu": m.mfu,
    "mbu": m.mbu,
    "ai": m.arithmetic_intensity,
    "compute_bound_prefill": m.is_prefill_compute_bound,
}, indent=2))
EOF

# ------- Step 5: Loadtest + calibrate --------------------------------------
log "Step 5/6: running loadtest ($CONCURRENCY concurrent × $NUM_REQUESTS requests)"
LOADTEST_JSON="$WORK_DIR/loadtest.json"
LOADTEST_ERR="$WORK_DIR/loadtest.err"
# Keep stdout (the --json blob) and stderr separate so stray log lines don't
# corrupt the JSON file. If loadtest exits nonzero, fail loud with both files.
python3 -m kv_planner.cli.main loadtest \
    --endpoint "http://127.0.0.1:$VLLM_PORT" \
    --model "$MODEL_HF" \
    --api openai \
    --concurrency "$CONCURRENCY" \
    --num-requests "$NUM_REQUESTS" \
    --num-predict "$OUTPUT_LEN" \
    --prompt "Write a detailed technical explanation of how attention mechanisms work in transformer models, covering scaled dot-product attention, queries, keys, values, multi-head attention, and grouped-query attention." \
    --json > "$LOADTEST_JSON" 2> "$LOADTEST_ERR" || {
        log "loadtest exited nonzero; stderr dump:"
        cat "$LOADTEST_ERR"
        log "loadtest stdout (first 40 lines):"
        head -40 "$LOADTEST_JSON" || true
    }
log "loadtest stdout size=$(wc -c <"$LOADTEST_JSON") bytes, first 2 lines:"
head -2 "$LOADTEST_JSON" || true

CALIBRATE_JSON="$WORK_DIR/calibrate.json"
CALIBRATE_ERR="$WORK_DIR/calibrate.err"
python3 -m kv_planner.cli.main calibrate \
    --endpoint "http://127.0.0.1:$VLLM_PORT" \
    --model "$MODEL_SLUG" \
    --gpu "$GPU_KEY" \
    --api openai \
    --concurrency 2 \
    --num-requests 8 \
    --num-predict 128 \
    --json > "$CALIBRATE_JSON" 2> "$CALIBRATE_ERR" || true

# ------- Step 6: Single JSON artifact ---------------------------------------
log "Step 6/6: writing validation_result.json"
python3 - <<EOF > "$WORK_DIR/validation_result.json"
import json, pathlib

def _load_or(path, default):
    p = pathlib.Path(path)
    if not p.exists():
        return default
    txt = p.read_text().strip()
    if not txt or not txt.startswith(("{", "[")):
        return default
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return default

prediction = _load_or("$PREDICTION_JSON", {})
roofline = _load_or("$WORK_DIR/roofline_raw.json", {})
loadtest = _load_or("$LOADTEST_JSON", {})
calibrate = _load_or("$CALIBRATE_JSON", {})

def _safe_percentile(d, metric, pct="p50", default=0.0):
    v = d.get(metric)
    if isinstance(v, dict):
        return v.get(pct) or default
    return default

measured_tpot_ms = _safe_percentile(loadtest, "tpot_ms", "p50", 0.0)
predicted_tpot_ms = roofline.get("tpot_ms", 0.0)

# MAPE on TPOT (best decode metric)
if measured_tpot_ms > 0 and predicted_tpot_ms > 0:
    mape_tpot = abs(predicted_tpot_ms - measured_tpot_ms) / measured_tpot_ms * 100
else:
    mape_tpot = None

# MAPE on aggregate throughput
measured_tps = loadtest.get("aggregate_tok_s", 0) or 0
predicted_tps = roofline.get("throughput_tok_s", 0.0)
if measured_tps > 0 and predicted_tps > 0:
    mape_tps = abs(predicted_tps - measured_tps) / measured_tps * 100
else:
    mape_tps = None

print(json.dumps({
    "config": {
        "gpu_key": "$GPU_KEY",
        "model_hf": "$MODEL_HF",
        "model_slug": "$MODEL_SLUG",
        "precision": "$PRECISION",
        "input_length": $INPUT_LEN,
        "output_length": $OUTPUT_LEN,
        "concurrency": $CONCURRENCY,
        "num_requests": $NUM_REQUESTS,
        "runtime": "vllm",
    },
    "predicted": roofline,
    "measured": {
        "aggregate_tok_s": measured_tps,
        "ttft_ms_p50": _safe_percentile(loadtest, "ttft_ms", "p50", None),
        "ttft_ms_p99": _safe_percentile(loadtest, "ttft_ms", "p99", None),
        "tpot_ms_p50": measured_tpot_ms,
        "tpot_ms_p99": _safe_percentile(loadtest, "tpot_ms", "p99", None),
        "e2e_ms_p50": _safe_percentile(loadtest, "e2e_ms", "p50", None),
        "e2e_ms_p99": _safe_percentile(loadtest, "e2e_ms", "p99", None),
        "errors": loadtest.get("errors", 0),
        "raw_keys": sorted(loadtest.keys()) if loadtest else [],
    },
    "calibration": {
        "derived_mbu": calibrate.get("calibrated_mbu"),
        "achieved_bandwidth_gb_s": calibrate.get("achieved_memory_bandwidth_gb_s"),
        "peak_bandwidth_gb_s": calibrate.get("peak_memory_bandwidth_gb_s"),
    },
    "accuracy": {
        "mape_tpot_pct": round(mape_tpot, 1) if mape_tpot is not None else None,
        "mape_throughput_pct": round(mape_tps, 1) if mape_tps is not None else None,
    },
}, indent=2))
EOF

cat "$WORK_DIR/validation_result.json"
log "Done. Artifact: $WORK_DIR/validation_result.json"
