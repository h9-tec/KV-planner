"""Training / fine-tuning capacity planner.

References:

* Rajbhandari et al., ZeRO (arxiv 1910.02054) — the 16 bytes/param AdamW
  accounting: 2 moments in fp32 (8 B) + fp32 master weights (4 B) +
  fp16 grad copy (2 B) + fp16 param copy (2 B) = 16 bytes/param for
  mixed-precision training.
* Korthikanti et al., Reducing Activation Recomputation (arxiv 2205.05198)
  — selective activation checkpointing math.
* Hu et al., LoRA (arxiv 2106.09685) — rank-decomposition trainable params.
* Dettmers et al., QLoRA (arxiv 2305.14314) — NF4-quantized backbone +
  LoRA adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kv_planner.domain import HardwareSpec, ModelConfig, PrecisionType, bytes_per_element

TrainingMethod = Literal["full_ft", "lora", "qlora"]


@dataclass(frozen=True)
class TrainingPlan:
    """Memory + compute estimates for one fine-tuning recipe."""

    method: TrainingMethod
    weight_precision: PrecisionType
    optimizer_bytes_per_param: float
    model_weight_gb: float
    gradient_gb: float
    optimizer_state_gb: float
    activation_gb: float
    total_memory_gb: float
    fits_per_gpu: bool
    # Compute
    tokens_per_step: int
    flops_per_step: float
    step_time_sec: float
    tokens_per_second: float
    est_training_hours: float
    est_cost_usd: float
    # LoRA-specific
    trainable_params: int


class TrainingPlanner:
    """Plan full-fine-tune / LoRA / QLoRA runs.

    Memory accounting (per GPU, assumes ZeRO-1 style sharding off by default):

    * full_ft fp16 mixed-precision:   **16 bytes/param** (AdamW + master + grad + weight copy)
    * LoRA (r=16, rank+alpha typical): **weights frozen** + small trainable set
    * QLoRA: **weights in NF4 (~4.5 bits/param)** + LoRA adapters

    Activation memory (transformer, with selective checkpointing):

        act_gb ≈ seq × batch × layers × hidden × (34 + ...some small terms)
                 × bpe / checkpoint_factor

    where ``checkpoint_factor`` is ~sqrt(layers) for "selective" and 1 for
    full-activation storage.
    """

    # Per-param bytes held on GPU for each training recipe (mixed-precision).
    # full_ft fp16: 2 (weights fp16) + 2 (grad fp16) + 4 (master fp32)
    #             + 4 (Adam m fp32) + 4 (Adam v fp32) = 16
    # full_ft bf16: same arithmetic
    # lora: weights frozen at 2 bytes + trainable adapter bytes (tiny)
    # qlora: weights in NF4 ≈ 0.6 bytes/param + adapter bytes

    PER_PARAM_BYTES = {
        "full_ft_fp16": 16.0,
        "full_ft_bf16": 16.0,
        "full_ft_fp32": 16.0,
        "lora_fp16": 2.0,        # frozen weights
        "lora_bf16": 2.0,
        "qlora": 0.6,            # NF4 quant
    }

    # Default LoRA rank that's ~standard for 7-13B fine-tunes
    DEFAULT_LORA_RANK = 16
    DEFAULT_LORA_ALPHA = 32
    # LoRA attaches by default to Q and V projections; some setups add K, O,
    # and MLP. Use "q_v" (~60 % of layers), or "all_linear" (~100 %).
    DEFAULT_LORA_TARGET = "q_v"

    def __init__(self, selective_checkpointing: bool = True) -> None:
        self._sac = selective_checkpointing

    # ---- Params accounting ------------------------------------------------
    def _trainable_params(
        self,
        model: ModelConfig,
        method: TrainingMethod,
        lora_rank: int,
        target: str,
    ) -> int:
        if method == "full_ft":
            return model.total_params()

        d = model.hidden_size
        d_kv = model.kv_hidden_size
        d_ff = model._ffn_intermediate
        layers = model.num_layers

        # LoRA params per targeted matrix = rank × (in + out)
        per_attn_pair = lora_rank * (d + d)           # Q/O projections
        per_kv_pair = lora_rank * (d + d_kv)          # K/V projections (GQA-aware)
        per_ffn_matrix = lora_rank * (d + d_ff)

        if target == "q_v":
            per_layer = 2 * per_attn_pair
        elif target == "qkvo":
            per_layer = 2 * per_attn_pair + 2 * per_kv_pair
        elif target == "all_linear":
            per_layer = (
                2 * per_attn_pair
                + 2 * per_kv_pair
                + model.ffn_num_matmuls * per_ffn_matrix
            )
        else:
            per_layer = 2 * per_attn_pair

        return per_layer * layers

    # ---- Activation memory (scales with batch × seq × layers × hidden) ----
    def _activation_gb(
        self,
        model: ModelConfig,
        batch_size: int,
        sequence_length: int,
        precision: PrecisionType,
    ) -> float:
        bpe = bytes_per_element(precision)
        # Memory-efficient attention (Flash) keeps activation per-layer ≈
        # batch · seq · hidden · bytes, with a constant factor from the
        # number of checkpointed tensors. The Korthikanti formula for SAC
        # comes to ~17 bytes-per-activation-element; we round to 20 for
        # safety with mixed-precision ops.
        per_layer_bytes = batch_size * sequence_length * model.hidden_size * bpe * 20
        total = per_layer_bytes * model.num_layers
        if self._sac:
            # Selective/activation checkpointing recomputes ~(sqrt(L)-1)
            # layers, storing only sqrt(L) — use a conservative 0.3× factor.
            total *= 0.30
        return total / 1e9

    # ---- Training-step FLOPs ----------------------------------------------
    @staticmethod
    def _flops_per_training_token(model: ModelConfig, sequence_length: int) -> float:
        """Forward + backward ≈ 3× forward. Forward roughly 2N + attn term."""
        from kv_planner.core.performance.roofline import flops_per_token_per_layer
        fwd_per_tok = model.num_layers * flops_per_token_per_layer(model, sequence_length)
        return 3.0 * fwd_per_tok  # forward + backward + optimizer

    # ---- Public API -------------------------------------------------------
    def plan(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        method: TrainingMethod = "lora",
        *,
        precision: PrecisionType = "bf16",
        batch_size: int = 4,
        sequence_length: int = 2048,
        num_epochs: int = 3,
        dataset_tokens: int = 1_000_000,
        lora_rank: int = DEFAULT_LORA_RANK,
        lora_target: str = DEFAULT_LORA_TARGET,
        gpu_hourly_cost: float | None = None,
    ) -> TrainingPlan:
        # --- Per-param byte cost -----------------------------------------
        if method == "full_ft":
            key = f"full_ft_{precision}"
        elif method == "lora":
            key = f"lora_{precision}"
        else:
            key = "qlora"
        per_param = self.PER_PARAM_BYTES.get(key, 16.0)

        total_params = model.total_params()
        trainable_params = self._trainable_params(model, method, lora_rank, lora_target)

        # Memory breakdown — per GPU (no TP sharding by default)
        weights_gb = total_params * per_param / 1e9  # includes grads + opt for full_ft
        if method != "full_ft":
            # LoRA/QLoRA: weights are frozen; trainable set needs its own
            # grads + optimizer. Trainable set is tiny vs total.
            weights_only_gb = weights_gb
            trainable_opt_gb = trainable_params * 16 / 1e9  # adapter AdamW
            weights_gb = weights_only_gb
            grad_gb = 0.0  # rolled into trainable_opt_gb
            opt_gb = trainable_opt_gb
        else:
            # For full_ft, PER_PARAM_BYTES=16 already includes grads + master + opt.
            grad_gb = total_params * 2 / 1e9  # just the tracked grad component
            opt_gb = total_params * 8 / 1e9   # Adam m + v in fp32

        activation_gb = self._activation_gb(model, batch_size, sequence_length, precision)
        total_gb = weights_gb + grad_gb + opt_gb + activation_gb
        # When method==full_ft, weights_gb already rolls up grads+opt; avoid
        # double count.
        if method == "full_ft":
            total_gb = weights_gb + activation_gb

        budget_gb = hardware.gpu_memory_gb * hardware.gpu_memory_utilization
        fits = total_gb <= budget_gb

        # --- Compute time -------------------------------------------------
        tokens_per_step = batch_size * sequence_length
        flops_per_token = self._flops_per_training_token(model, sequence_length)
        flops_per_step = flops_per_token * tokens_per_step

        peak_tflops = hardware.peak_tflops_for(precision)
        mfu = 0.35  # realistic training MFU: 30-45 % on H100
        achievable_tflops = peak_tflops * mfu
        step_time_sec = flops_per_step / (achievable_tflops * 1e12)

        total_tokens = dataset_tokens * num_epochs
        total_steps = total_tokens / tokens_per_step
        total_compute_sec = total_steps * step_time_sec
        total_hours = total_compute_sec / 3600

        tokens_per_second = tokens_per_step / step_time_sec if step_time_sec > 0 else 0

        # --- Cost ---------------------------------------------------------
        rate = gpu_hourly_cost
        if rate is None:
            rate = hardware.gpu_hourly_cost or _default_rate(hardware.gpu_model)
        est_cost = total_hours * hardware.num_gpus * rate

        return TrainingPlan(
            method=method,
            weight_precision=precision,
            optimizer_bytes_per_param=per_param,
            model_weight_gb=weights_gb,
            gradient_gb=grad_gb,
            optimizer_state_gb=opt_gb,
            activation_gb=activation_gb,
            total_memory_gb=total_gb,
            fits_per_gpu=fits,
            tokens_per_step=tokens_per_step,
            flops_per_step=flops_per_step,
            step_time_sec=step_time_sec,
            tokens_per_second=tokens_per_second,
            est_training_hours=total_hours,
            est_cost_usd=est_cost,
            trainable_params=trainable_params,
        )


def _default_rate(gpu_model: str) -> float:
    """Coarse fallback $/hr when hardware has no price set."""
    rates = {
        "H100-SXM-80GB": 4.50, "H200-SXM-141GB": 6.00, "A100-SXM-80GB": 2.50,
        "A100-SXM-40GB": 1.80, "RTX-4090": 0.21, "RTX-5090": 0.35,
        "RTX-3090": 0.12, "MI300X": 5.50,
    }
    return rates.get(gpu_model, 1.00)
