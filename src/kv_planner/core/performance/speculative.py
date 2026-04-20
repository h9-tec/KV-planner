"""Speculative decoding — draft-model physics.

Models EAGLE-3 / Medusa / Lookahead / vanilla-draft as a two-model pipeline:

* **Target** model does verification (full forward pass per verified step,
  but over K draft tokens in parallel — compute-bound if K is large).
* **Draft** model generates K candidate tokens auto-regressively (K memory-
  bound steps at low cost because the draft is tiny).
* Acceptance rate α ∈ [0, 1] governs effective tokens per verified step:
  E[accepted] = (1 − α^(K+1)) / (1 − α)    — the standard geometric-series
  formula (Leviathan et al. 2022, arxiv 2211.17192).

Net speedup versus target-alone decode:

    speedup = E[accepted] / (1 + K · (draft_cost / target_cost))

where draft_cost / target_cost ≈ draft_params / target_params (both decode
is memory-bound; bytes scale with params).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpecMethod = Literal["eagle3", "medusa", "lookahead", "draft_model", "none"]


# Published acceptance rates (chat / code / math mix) from each paper.
#   EAGLE-3: 0.80  (NeurIPS'25, https://sites.google.com/view/eagle-llm)
#   Medusa:  0.65  (Cai et al. 2024, arxiv 2401.10774)
#   Lookahead: 0.55 (no-train)
#   Draft-model (a small LM, e.g. Llama-3.2-1B for Llama-3-8B target): 0.70
_DEFAULT_ACCEPTANCE: dict[SpecMethod, float] = {
    "eagle3": 0.80,
    "medusa": 0.65,
    "lookahead": 0.55,
    "draft_model": 0.70,
    "none": 0.0,
}

# Default speculation window K.
_DEFAULT_K: dict[SpecMethod, int] = {
    "eagle3": 6,
    "medusa": 4,
    "lookahead": 5,
    "draft_model": 4,
    "none": 0,
}


@dataclass(frozen=True)
class SpecDecodePlan:
    method: SpecMethod
    K: int
    acceptance_rate: float
    draft_cost_ratio: float
    expected_tokens_per_verify_step: float
    speedup: float
    effective_tpot_ms: float       # decode latency per output token, with spec
    draft_kv_bytes_per_token: int  # additional KV cost (draft)

    @property
    def percent_faster(self) -> float:
        return (self.speedup - 1.0) * 100


def expected_accepted_tokens(accept_rate: float, K: int) -> float:
    """Geometric-series expectation over K speculative tokens.

    If α = 1 every token is accepted and the sum collapses to K+1.
    If α = 0 we always fall back to the 1 forced target token.
    """
    if accept_rate >= 1.0:
        return float(K + 1)
    if accept_rate <= 0.0:
        return 1.0
    return (1.0 - accept_rate ** (K + 1)) / (1.0 - accept_rate)


def speedup_over_autoregressive(
    accept_rate: float,
    K: int,
    draft_cost_ratio: float,
) -> float:
    """Net speedup vs single-target autoregressive decode.

    ``draft_cost_ratio`` is (draft_params / target_params) for memory-bound
    decode where per-token latency ≈ bytes / bw. EAGLE/Medusa share the
    target's hidden state so the ratio is effectively a small head (~1–3 %).
    """
    if accept_rate <= 0 or K <= 0:
        return 1.0
    accepted = expected_accepted_tokens(accept_rate, K)
    verify_overhead = 1.0 + K * max(0.0, draft_cost_ratio)
    return accepted / verify_overhead


def plan(
    method: SpecMethod,
    target_model_params: int,
    draft_model_params: int,
    target_tpot_ms: float,
    target_kv_bytes_per_token: int,
    K: int | None = None,
    acceptance_rate: float | None = None,
) -> SpecDecodePlan:
    """Build a :class:`SpecDecodePlan` for given target + draft."""
    if method == "none" or target_model_params <= 0:
        return SpecDecodePlan(
            method="none", K=0, acceptance_rate=0.0,
            draft_cost_ratio=0.0, expected_tokens_per_verify_step=1.0,
            speedup=1.0, effective_tpot_ms=target_tpot_ms,
            draft_kv_bytes_per_token=0,
        )

    K = K if K is not None else _DEFAULT_K[method]
    alpha = acceptance_rate if acceptance_rate is not None else _DEFAULT_ACCEPTANCE[method]

    # EAGLE / Medusa are "shared hidden state" — draft is a small head, ratio is
    # essentially size_of_head / size_of_target ≈ 0.01–0.03.
    if method in ("eagle3", "medusa"):
        draft_ratio = 0.02
    elif method == "lookahead":
        # No extra model; cost is CPU verification work. Estimate 1% overhead.
        draft_ratio = 0.01
    else:  # draft_model
        draft_ratio = (
            draft_model_params / target_model_params if target_model_params > 0 else 0.1
        )

    accepted = expected_accepted_tokens(alpha, K)
    speedup = speedup_over_autoregressive(alpha, K, draft_ratio)
    effective_tpot = target_tpot_ms / speedup if speedup > 0 else target_tpot_ms

    # Draft KV = (draft_params / target_params) × target_kv (rough approx).
    draft_kv = int(target_kv_bytes_per_token * draft_ratio)

    return SpecDecodePlan(
        method=method,
        K=K,
        acceptance_rate=alpha,
        draft_cost_ratio=draft_ratio,
        expected_tokens_per_verify_step=accepted,
        speedup=speedup,
        effective_tpot_ms=effective_tpot,
        draft_kv_bytes_per_token=draft_kv,
    )
