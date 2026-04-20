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
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet vllm

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
    if [ "$i" = "90" ]; then
        log "vLLM failed to start. Last 20 log lines:"
        tail -20 "$VLLM_LOG"
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
python3 -m kv_planner.cli.main loadtest \
    --endpoint "http://127.0.0.1:$VLLM_PORT" \
    --model "$MODEL_HF" \
    --api openai \
    --concurrency "$CONCURRENCY" \
    --num-requests "$NUM_REQUESTS" \
    --num-predict "$OUTPUT_LEN" \
    --prompt "Write a detailed technical explanation of how attention mechanisms work in transformer models, covering scaled dot-product attention, queries, keys, values, multi-head attention, and grouped-query attention." \
    --json > "$LOADTEST_JSON" 2>&1

CALIBRATE_JSON="$WORK_DIR/calibrate.json"
python3 -m kv_planner.cli.main calibrate \
    --endpoint "http://127.0.0.1:$VLLM_PORT" \
    --model "$MODEL_SLUG" \
    --gpu "$GPU_KEY" \
    --api openai \
    --concurrency 2 \
    --num-requests 8 \
    --num-predict 128 \
    --json > "$CALIBRATE_JSON" 2>&1 || true

# ------- Step 6: Single JSON artifact ---------------------------------------
log "Step 6/6: writing validation_result.json"
python3 - <<EOF > "$WORK_DIR/validation_result.json"
import json

prediction = json.loads(open("$PREDICTION_JSON").read()) if open("$PREDICTION_JSON").read().strip().startswith("{") else {}
roofline = json.loads(open("$WORK_DIR/roofline_raw.json").read())
loadtest = json.loads(open("$LOADTEST_JSON").read())
try:
    calibrate = json.loads(open("$CALIBRATE_JSON").read())
except Exception:
    calibrate = {}

measured_tpot_ms = loadtest.get("tpot_ms", {}).get("p50", 0.0)
predicted_tpot_ms = roofline["tpot_ms"]

# MAPE on TPOT (best decode metric)
if measured_tpot_ms > 0 and predicted_tpot_ms > 0:
    mape_tpot = abs(predicted_tpot_ms - measured_tpot_ms) / measured_tpot_ms * 100
else:
    mape_tpot = None

# MAPE on aggregate throughput
measured_tps = loadtest.get("aggregate_tok_s", 0)
predicted_tps = roofline["throughput_tok_s"]
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
        "ttft_ms_p50": loadtest.get("ttft_ms", {}).get("p50"),
        "ttft_ms_p99": loadtest.get("ttft_ms", {}).get("p99"),
        "tpot_ms_p50": measured_tpot_ms,
        "tpot_ms_p99": loadtest.get("tpot_ms", {}).get("p99"),
        "e2e_ms_p50": loadtest.get("e2e_ms", {}).get("p50"),
        "e2e_ms_p99": loadtest.get("e2e_ms", {}).get("p99"),
        "errors": loadtest.get("errors", 0),
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
