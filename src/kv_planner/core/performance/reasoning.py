"""Reasoning-model KV planner.

Unlike chat, reasoning models (o1/o3/DeepSeek-R1/QwQ/Kimi K2 reasoning) emit
**thinking tokens** that can be 5–20× the final-answer length. DeepSeek-R1
averages 23k tokens/AIME question (2506); the distribution is heavy-tailed,
so p99 is what must fit in VRAM — not the mean.

We model thinking-token output as a log-normal distribution and compute the
KV working-set at p99 context length: ``kv_per_token × (prompt + p99_think +
answer)``. The number to plan for is NOT mean memory; it's p99.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kv_planner.domain import ModelConfig, PrecisionType, bytes_per_element


@dataclass(frozen=True)
class ReasoningWorkload:
    """Log-normal thinking-token distribution + fixed answer length.

    Known points (published in the R1 paper, QwQ report, OpenAI o1 preview
    cost disclosures):

    * ``think_mean``: average thinking tokens across the task mix
    * ``think_p99``: upper tail (what we must fit)
    * ``answer_tokens``: final visible answer
    """

    think_mean: int = 2000
    think_p99: int = 16000
    answer_tokens: int = 400


# A few published profiles — users can override with their own.
PROFILES: dict[str, ReasoningWorkload] = {
    # DeepSeek-R1 on AIME / MATH — avg ~23k thinking tokens on hard math.
    "deepseek-r1-math": ReasoningWorkload(
        think_mean=12000, think_p99=32000, answer_tokens=400,
    ),
    # o3-mini on chat + code assistance — much shorter.
    "o3-mini-chat": ReasoningWorkload(
        think_mean=1500, think_p99=6000, answer_tokens=300,
    ),
    # QwQ-32B on coding tasks.
    "qwq-code": ReasoningWorkload(
        think_mean=4000, think_p99=12000, answer_tokens=500,
    ),
    # A balanced mix (reasoning benchmark suite).
    "balanced": ReasoningWorkload(
        think_mean=3000, think_p99=12000, answer_tokens=400,
    ),
}


@dataclass(frozen=True)
class ReasoningPlan:
    model: str
    workload: ReasoningWorkload
    precision: PrecisionType
    prompt_tokens: int
    kv_bytes_mean_per_seq: int
    kv_bytes_p99_per_seq: int
    kv_gb_mean_batch: float
    kv_gb_p99_batch: float
    batch_size: int
    p99_context_tokens: int

    @property
    def p99_over_mean_ratio(self) -> float:
        return (
            self.kv_bytes_p99_per_seq / self.kv_bytes_mean_per_seq
            if self.kv_bytes_mean_per_seq > 0 else 0.0
        )


def plan_reasoning(
    model: ModelConfig,
    workload: ReasoningWorkload,
    *,
    prompt_tokens: int = 500,
    batch_size: int = 1,
    precision: PrecisionType = "fp16",
) -> ReasoningPlan:
    """Compute mean + p99 KV footprint for a reasoning workload."""
    kv_per_tok = model.kv_cache_bytes_per_token(precision)

    mean_ctx = prompt_tokens + workload.think_mean + workload.answer_tokens
    p99_ctx = prompt_tokens + workload.think_p99 + workload.answer_tokens

    kv_mean = kv_per_tok * mean_ctx
    kv_p99 = kv_per_tok * p99_ctx

    return ReasoningPlan(
        model=model.name,
        workload=workload,
        precision=precision,
        prompt_tokens=prompt_tokens,
        kv_bytes_mean_per_seq=kv_mean,
        kv_bytes_p99_per_seq=kv_p99,
        kv_gb_mean_batch=kv_mean * batch_size / 1e9,
        kv_gb_p99_batch=kv_p99 * batch_size / 1e9,
        batch_size=batch_size,
        p99_context_tokens=p99_ctx,
    )
