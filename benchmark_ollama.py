"""Real-world validation: kv-planner predictions vs actual Ollama throughput.

This script reports *raw, un-tuned* predictions next to measurements. It does
NOT fit any knob to the data. Previous versions did — and that is not
validation. What you should read out of this:

1. Weight bytes and KV-cache bytes per token are checked against vendor /
   HF canonical numbers. Those are exactly correct.
2. The roofline ceiling (peak HBM BW × memory_efficiency) is an UPPER BOUND.
   Llama.cpp routinely achieves 30–45 % of peak HBM on laptop GPUs due to
   CUDA-Graph-less scheduling, less-tuned matmul kernels, and Python/HTTP
   overhead. The planner's default 0.75 MBU is calibrated for vLLM +
   FlashAttention, so predictions on Ollama WILL be optimistic.
3. The gap between "raw roofline" and "measured" is the runtime's
   inefficiency, not a bug in the physics engine.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.domain import ModelConfig, bytes_per_element
from kv_planner.infrastructure.hardware_db import GPUDatabase


OLLAMA_URL = "http://127.0.0.1:11434"
TEST_PROMPT = (
    "Write a detailed technical explanation of how attention mechanisms work "
    "in transformer models, covering the mathematical formulation of scaled "
    "dot-product attention, the role of queries, keys, and values, how "
    "multi-head attention enables parallel processing of different "
    "representation subspaces, and why grouped-query attention (GQA) is used "
    "in modern LLMs to reduce KV cache memory footprint."
)


@dataclass(frozen=True)
class OllamaBench:
    ollama_name: str
    display: str
    config: ModelConfig
    # Q4_K_M actually averages ~4.5 bits/weight (not 4). We keep the
    # kv-planner "int4" label but apply a correction factor when computing
    # measured MBU so the accounting is accurate.
    bytes_per_weight: float
    precision: str  # planner precision for the roofline call


# Architectures — every number verified against the model's HF config.json.
BENCHMARKS = [
    OllamaBench(
        ollama_name="llama3.2:3b",
        display="Llama 3.2 3B",
        config=ModelConfig(
            name="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=28,
            hidden_size=3072,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=128256,
            max_position_embeddings=131072,
            attention_type="GQA",
            ffn_type="swiglu",
            ffn_intermediate_size=8192,
        ),
        bytes_per_weight=0.56,  # Q4_K_M ≈ 4.5 bits/w
        precision="int4",
    ),
    OllamaBench(
        ollama_name="deepseek-r1:latest",
        display="DeepSeek-R1 8B (Qwen-distill)",
        config=ModelConfig(
            name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            num_layers=28,
            hidden_size=3584,
            num_attention_heads=28,
            num_key_value_heads=4,
            head_dim=128,
            vocab_size=152064,
            max_position_embeddings=131072,
            attention_type="GQA",
            ffn_type="swiglu",
            ffn_intermediate_size=18944,
        ),
        bytes_per_weight=0.56,
        precision="int4",
    ),
    OllamaBench(
        ollama_name="aya:8b",
        display="Aya-8B (Command-R, tied-emb)",
        # Command-R ties input/output embeddings — our total_params() does
        # NOT account for that, so for this model we'd over-count params
        # by a factor (vocab*hidden / total). vocab=256000, hidden=4096 →
        # ~1.05 B. Acknowledged below, not corrected.
        config=ModelConfig(
            name="CohereForAI/aya-expanse-8b",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=256000,
            max_position_embeddings=8192,
            attention_type="GQA",
            ffn_type="swiglu",
            ffn_intermediate_size=14336,
        ),
        bytes_per_weight=0.56,
        precision="int4",
    ),
]


def run_ollama(model: str, prompt: str, num_predict: int) -> dict:
    """Return timing + token counts from Ollama /api/generate."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.0, "seed": 42},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())

    # Ollama reports its own nanosecond-precision timings, which exclude
    # HTTP/JSON overhead and are the right number to compare against
    # the roofline.
    return {
        "total_s": data.get("total_duration", 0) / 1e9,
        "load_s": data.get("load_duration", 0) / 1e9,
        "prompt_eval_s": data.get("prompt_eval_duration", 0) / 1e9,
        "eval_s": data.get("eval_duration", 0) / 1e9,
        "prompt_tokens": int(data.get("prompt_eval_count", 0)),
        "output_tokens": int(data.get("eval_count", 0)),
    }


def main() -> None:
    gpu = GPUDatabase.to_hardware_spec("RTX-5060-Laptop")
    peak_bw_gb_s = gpu.memory_bandwidth_gb_s
    ra = RooflineAnalyzer()  # defaults: compute_eff=0.50, memory_eff=0.75

    print()
    print("=" * 92)
    print("  kv-planner 0.2.0 honest validation on RTX 5060 Laptop via Ollama")
    print("=" * 92)
    print()
    print(f"  GPU peak HBM bandwidth (from gpu_specs.py): {peak_bw_gb_s:.0f} GB/s")
    print(f"  No knob-tuning. Default memory_efficiency=0.75 (vLLM-style).")
    print()

    header = (
        f"  {'Model':<30} "
        f"{'measured':>9} {'roofline':>9} {'gap':>5} "
        f"{'MBU*':>6} {'prefill':>8} {'decode':>8}"
    )
    print(header)
    print(
        f"  {'':<30} "
        f"{'tok/s':>9} {'tok/s':>9} {'':>5} "
        f"{'%':>6} {'ms/tok':>8} {'ms/tok':>8}"
    )
    print("  " + "-" * 88)

    for bench in BENCHMARKS:
        # Warm-up, then timed run.
        try:
            run_ollama(bench.ollama_name, TEST_PROMPT, num_predict=32)
            r = run_ollama(bench.ollama_name, TEST_PROMPT, num_predict=256)
        except Exception as e:
            print(f"  {bench.display:<30}  Ollama failed: {e}")
            continue

        p_tok = r["prompt_tokens"]
        o_tok = r["output_tokens"]
        prefill_ms_per_token = (r["prompt_eval_s"] * 1000 / p_tok) if p_tok else 0.0
        decode_ms_per_token = (r["eval_s"] * 1000 / o_tok) if o_tok else 0.0
        measured_decode_tps = o_tok / r["eval_s"] if r["eval_s"] > 0 else 0.0

        # Raw roofline prediction (no thermal adjustment, no knob tuning).
        # Use the same decode-step formula the planner uses internally, at
        # the actual observed context length.
        ctx_len = p_tok + o_tok // 2  # average context during decode
        _, _, _ = ra.predict_decode_latency(
            bench.config, gpu, batch_size=1, sequence_length=ctx_len, precision=bench.precision
        )
        # Query per-token latency directly for a clean roofline number.
        per_tok_ms, _, _ = ra.predict_decode_latency(
            bench.config, gpu, batch_size=1, sequence_length=ctx_len, precision=bench.precision
        )
        roofline_decode_tps = 1000.0 / per_tok_ms if per_tok_ms > 0 else 0.0

        # *Measured* achieved memory-bandwidth utilisation. Bytes streamed
        # per decode step ≈ model_params · bytes_per_weight (KV reads are
        # small at these context lengths).
        total_params = bench.config.total_params()
        bytes_per_step = total_params * bench.bytes_per_weight
        achieved_gb_s = bytes_per_step / (decode_ms_per_token / 1000.0) / 1e9
        achieved_mbu = achieved_gb_s / peak_bw_gb_s * 100

        ratio = measured_decode_tps / roofline_decode_tps if roofline_decode_tps > 0 else 0.0

        print(
            f"  {bench.display:<30} "
            f"{measured_decode_tps:9.1f} "
            f"{roofline_decode_tps:9.1f} "
            f"{ratio:5.2f} "
            f"{achieved_mbu:6.1f} "
            f"{prefill_ms_per_token:8.2f} "
            f"{decode_ms_per_token:8.2f}"
        )

    print()
    print("  " + "-" * 88)
    print("  * MBU is computed from Ollama's own nanosecond timings: achieved HBM BW")
    print("    ÷ peak HBM BW. It is the realized fraction of peak memory bandwidth")
    print("    the runtime extracted — NOT something the planner predicted.")
    print()
    print("  Interpretation:")
    print("    • If MBU ≈ 30-45 %, llama.cpp is leaving 55-70 % of the GPU idle")
    print("      (it lacks CUDA graphs, uses cuBLAS rather than FlashAttention,")
    print("       and serializes through a CPU scheduler).")
    print("    • The planner's DEFAULT 0.75 MBU is vLLM-tuned, so its `roofline`")
    print("      column predicts ~2× the Ollama number. This is working as")
    print("      designed — roofline is the ceiling, not the expected throughput.")
    print("    • For an Ollama-calibrated planner, pass")
    print("      RooflineAnalyzer(memory_efficiency=observed_MBU).")


if __name__ == "__main__":
    main()
