"""Unit tests for loadtest — aggregator + SLO + knee detection.

The driver (_one_ollama / _one_openai) is an integration surface; it's
exercised end-to-end by benchmark_ollama.py. Here we test the pure-Python
aggregator logic and the SLO evaluator.
"""

from __future__ import annotations

import pytest

from kv_planner.loadtest.runner import (
    LoadTestResult,
    RequestResult,
    SloTargets,
    _aggregate,
    _knee_of,
    _quantiles,
)


def _mk(ok=True, ttft=0.1, tpot=0.01, total=1.0, out=100) -> RequestResult:
    return RequestResult(ok=ok, ttft_s=ttft, tpot_s=tpot, total_s=total,
                         prompt_tokens=50, output_tokens=out)


def test_quantiles_handles_empty():
    assert _quantiles([]) == (0.0, 0.0, 0.0)


def test_quantiles_returns_sorted_percentiles():
    p50, p95, p99 = _quantiles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert p50 <= p95 <= p99
    assert p95 >= 9.0  # near the top


def test_aggregate_counts_successes_and_errors():
    rs = [_mk(ok=True) for _ in range(8)] + [_mk(ok=False) for _ in range(2)]
    agg = _aggregate(
        endpoint="ep", model="m", api="ollama", concurrency=4, num_requests=10,
        wall_s=10.0, per_request=rs, slo=None,
    )
    assert agg.error_count == 2
    assert agg.total_output_tokens == 800
    assert agg.aggregate_tok_s == pytest.approx(80.0)


def test_slo_passes_only_when_all_three_thresholds_met():
    slo = SloTargets(ttft_ms=200, tpot_ms=20, e2e_ms=2000)
    passing = _mk(ttft=0.1, tpot=0.01, total=1.0)       # ttft=100ms, tpot=10ms, e2e=1000ms
    ttft_fail = _mk(ttft=0.3, tpot=0.01, total=1.0)
    tpot_fail = _mk(ttft=0.1, tpot=0.03, total=1.0)
    e2e_fail = _mk(ttft=0.1, tpot=0.01, total=3.0)
    failed = _mk(ok=False)

    assert slo.passes(passing)
    assert not slo.passes(ttft_fail)
    assert not slo.passes(tpot_fail)
    assert not slo.passes(e2e_fail)
    assert not slo.passes(failed)


def test_goodput_percent_is_accurate():
    slo = SloTargets(ttft_ms=200, e2e_ms=2000)
    rs = (
        [_mk(ttft=0.1, total=1.0) for _ in range(7)]          # pass
        + [_mk(ttft=0.3, total=1.0) for _ in range(3)]        # miss TTFT
    )
    agg = _aggregate("ep", "m", "ollama", 4, 10, 5.0, rs, slo)
    assert agg.pass_count == 7
    assert agg.goodput_pct == pytest.approx(70.0)


def test_knee_identifies_saturation():
    # Synthetic: throughput grows strongly from c=1 to c=2 then flattens.
    pts = [
        LoadTestResult("e", "m", "ollama", 1, 10, 1.0, [], aggregate_tok_s=100),
        LoadTestResult("e", "m", "ollama", 2, 10, 1.0, [], aggregate_tok_s=180),  # +80%
        LoadTestResult("e", "m", "ollama", 4, 10, 1.0, [], aggregate_tok_s=186),  # +3%
        LoadTestResult("e", "m", "ollama", 8, 10, 1.0, [], aggregate_tok_s=190),  # +2%
    ]
    # From c=2 onward, gain < 10% → knee at concurrency=2
    assert _knee_of(pts) == 2


def test_knee_on_ever_growing_throughput_returns_max():
    pts = [
        LoadTestResult("e", "m", "ollama", 1, 10, 1.0, [], aggregate_tok_s=100),
        LoadTestResult("e", "m", "ollama", 2, 10, 1.0, [], aggregate_tok_s=200),
        LoadTestResult("e", "m", "ollama", 4, 10, 1.0, [], aggregate_tok_s=400),
    ]
    assert _knee_of(pts) == 4
