"""Threaded concurrent load tester with per-token timing.

Supports two API flavours:

* ``ollama`` — POST ``/api/generate``, streamed JSON-lines response; each
  line has a ``response`` chunk + final object with timings. No TTFT token
  stream needed from us, Ollama reports its own nanosecond precision.
* ``openai`` — POST ``/v1/chat/completions`` with ``stream=True``, SSE
  ``data: {...}\\n\\n`` chunks; we measure TTFT from first byte of the
  response body, TPOT from inter-chunk deltas.

For each request we record:
* ``ttft_s`` — time to first token (s)
* ``tpot_s`` — average inter-token latency (s) across the generated tokens
* ``total_s`` — end-to-end wall clock (s)
* ``output_tokens`` — measured token count (Ollama) or estimated (OpenAI)
* ``prompt_tokens`` — from the API response when available
* ``ok`` — success flag

Aggregates report p50/p95/p99/mean/max for each metric, plus **goodput**:
the fraction of requests that meet the joint SLO (TTFT ≤ X and TPOT ≤ Y
and E2E ≤ Z). Goodput is the 2026 serving-SLO consensus metric.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from http.client import HTTPConnection
from typing import Callable, Literal, Optional

ApiFlavour = Literal["ollama", "openai"]


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    ttft_s: float
    tpot_s: float        # mean inter-token latency
    total_s: float
    prompt_tokens: int
    output_tokens: int
    error: str = ""

    @property
    def throughput_tok_s(self) -> float:
        return self.output_tokens / self.total_s if self.total_s > 0 else 0.0


@dataclass
class SloTargets:
    """Optional SLOs for goodput accounting."""

    ttft_ms: Optional[float] = None    # p99 TTFT target
    tpot_ms: Optional[float] = None    # p99 inter-token target
    e2e_ms: Optional[float] = None     # p99 end-to-end target

    def passes(self, r: RequestResult) -> bool:
        if not r.ok:
            return False
        if self.ttft_ms is not None and r.ttft_s * 1000 > self.ttft_ms:
            return False
        if self.tpot_ms is not None and r.tpot_s * 1000 > self.tpot_ms:
            return False
        if self.e2e_ms is not None and r.total_s * 1000 > self.e2e_ms:
            return False
        return True


@dataclass(frozen=True)
class LoadTestResult:
    endpoint: str
    model: str
    api: ApiFlavour
    concurrency: int
    num_requests: int
    wall_s: float
    per_request: list[RequestResult]
    slo: Optional[SloTargets] = None

    # Aggregates (filled in post hoc)
    pass_count: int = 0
    error_count: int = 0
    goodput_pct: float = 0.0

    # Latency distributions (s) — None if no successful requests
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0
    e2e_p50: float = 0.0
    e2e_p95: float = 0.0
    e2e_p99: float = 0.0

    total_output_tokens: int = 0
    aggregate_tok_s: float = 0.0
    mean_per_request_tok_s: float = 0.0
    # Concurrent achieved TPOT (mean across the run)
    avg_tpot_ms: float = 0.0


@dataclass(frozen=True)
class SweepResult:
    endpoint: str
    model: str
    points: list[LoadTestResult]   # one per concurrency level
    knee_concurrency: Optional[int] = None


# ---------------------------------------------------------------------------
# Request drivers
# ---------------------------------------------------------------------------


def _one_ollama(
    host: str, port: int, model: str, prompt: str, num_predict: int, timeout: float,
) -> RequestResult:
    """Call Ollama /api/generate with stream=True and measure per-token timing."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict, "temperature": 0.0, "seed": 42},
    }).encode()
    conn = HTTPConnection(host, port, timeout=timeout)
    try:
        t_start = time.perf_counter()
        conn.request(
            "POST", "/api/generate",
            body=body, headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        t_first: Optional[float] = None
        last_chunk_t: Optional[float] = None
        per_token_deltas: list[float] = []
        prompt_tokens = 0
        output_tokens = 0
        # Ollama streams newline-delimited JSON.
        buf = b""
        while True:
            chunk = resp.read1(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                if obj.get("response") and t_first is None:
                    t_first = now
                if obj.get("response"):
                    if last_chunk_t is not None:
                        per_token_deltas.append(now - last_chunk_t)
                    last_chunk_t = now
                if obj.get("done"):
                    prompt_tokens = int(obj.get("prompt_eval_count", 0))
                    output_tokens = int(obj.get("eval_count", 0))
        t_end = time.perf_counter()

        if output_tokens == 0:
            # Fallback: count response bytes isn't a token count, but at least
            # signal that something came back.
            output_tokens = len(per_token_deltas)

        ttft = (t_first - t_start) if t_first else (t_end - t_start)
        tpot = statistics.mean(per_token_deltas) if per_token_deltas else 0.0
        return RequestResult(
            ok=True,
            ttft_s=ttft, tpot_s=tpot, total_s=t_end - t_start,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        )
    except Exception as e:
        return RequestResult(
            ok=False, ttft_s=0.0, tpot_s=0.0, total_s=0.0,
            prompt_tokens=0, output_tokens=0, error=str(e)[:200],
        )
    finally:
        conn.close()


def _one_openai(
    host: str, port: int, scheme: str, path: str,
    api_key: Optional[str],
    model: str, prompt: str, num_predict: int, timeout: float,
) -> RequestResult:
    """Call an OpenAI-compatible /v1/chat/completions endpoint with streaming."""
    from http.client import HTTPSConnection
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_predict,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    Conn = HTTPSConnection if scheme == "https" else HTTPConnection
    conn = Conn(host, port, timeout=timeout)
    try:
        t_start = time.perf_counter()
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status >= 400:
            err = resp.read(512).decode(errors="replace")
            return RequestResult(
                ok=False, ttft_s=0.0, tpot_s=0.0, total_s=time.perf_counter() - t_start,
                prompt_tokens=0, output_tokens=0, error=f"HTTP {resp.status}: {err}",
            )
        t_first: Optional[float] = None
        last_chunk_t: Optional[float] = None
        deltas: list[float] = []
        output_tokens = 0
        prompt_tokens = 0
        buf = b""
        while True:
            chunk = resp.read1(4096)
            if not chunk:
                break
            buf += chunk
            # SSE frames are separated by \n\n
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                for raw in frame.splitlines():
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    now = time.perf_counter()
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content"):
                        if t_first is None:
                            t_first = now
                        if last_chunk_t is not None:
                            deltas.append(now - last_chunk_t)
                        last_chunk_t = now
                        output_tokens += 1
                    if obj.get("usage"):
                        prompt_tokens = int(obj["usage"].get("prompt_tokens", 0))
                        output_tokens = int(obj["usage"].get("completion_tokens", output_tokens))
        t_end = time.perf_counter()
        ttft = (t_first - t_start) if t_first else (t_end - t_start)
        tpot = statistics.mean(deltas) if deltas else 0.0
        return RequestResult(
            ok=True,
            ttft_s=ttft, tpot_s=tpot, total_s=t_end - t_start,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        )
    except Exception as e:
        return RequestResult(
            ok=False, ttft_s=0.0, tpot_s=0.0, total_s=0.0,
            prompt_tokens=0, output_tokens=0, error=str(e)[:200],
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _quantiles(xs: list[float]) -> tuple[float, float, float]:
    """Return (p50, p95, p99). Empty → (0, 0, 0)."""
    if not xs:
        return 0.0, 0.0, 0.0
    xs = sorted(xs)
    n = len(xs)
    def q(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs[idx]
    return q(0.50), q(0.95), q(0.99)


def _aggregate(
    endpoint: str, model: str, api: ApiFlavour, concurrency: int,
    num_requests: int, wall_s: float,
    per_request: list[RequestResult], slo: Optional[SloTargets],
) -> LoadTestResult:
    ok = [r for r in per_request if r.ok]
    ttft = [r.ttft_s for r in ok]
    tpot = [r.tpot_s for r in ok if r.tpot_s > 0]
    e2e = [r.total_s for r in ok]
    total_out = sum(r.output_tokens for r in ok)
    t50, t95, t99 = _quantiles(ttft)
    p50, p95, p99 = _quantiles(tpot)
    e50, e95, e99 = _quantiles(e2e)
    pass_count = sum(1 for r in per_request if (slo or SloTargets()).passes(r))
    good = (pass_count / num_requests * 100) if num_requests > 0 else 0.0
    mean_per_req = (
        statistics.mean([r.throughput_tok_s for r in ok]) if ok else 0.0
    )
    return LoadTestResult(
        endpoint=endpoint, model=model, api=api,
        concurrency=concurrency, num_requests=num_requests, wall_s=wall_s,
        per_request=per_request, slo=slo,
        pass_count=pass_count,
        error_count=num_requests - len(ok),
        goodput_pct=good,
        ttft_p50=t50, ttft_p95=t95, ttft_p99=t99,
        tpot_p50=p50, tpot_p95=p95, tpot_p99=p99,
        e2e_p50=e50, e2e_p95=e95, e2e_p99=e99,
        total_output_tokens=total_out,
        aggregate_tok_s=total_out / wall_s if wall_s > 0 else 0.0,
        mean_per_request_tok_s=mean_per_req,
        avg_tpot_ms=(statistics.mean(tpot) * 1000) if tpot else 0.0,
    )


# ---------------------------------------------------------------------------
# LoadTester
# ---------------------------------------------------------------------------


class LoadTester:
    def __init__(self) -> None:
        pass

    def run(
        self,
        endpoint: str,
        model: str,
        prompt: str = "Explain attention mechanisms in transformers.",
        *,
        api: ApiFlavour = "ollama",
        concurrency: int = 8,
        num_requests: int = 32,
        num_predict: int = 128,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        slo: Optional[SloTargets] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> LoadTestResult:
        """Fire ``num_requests`` at ``endpoint`` with ``concurrency`` in flight."""
        parsed = urllib.parse.urlparse(endpoint)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if scheme == "https" else 11434 if api == "ollama" else 8000)
        path = parsed.path or ("/api/generate" if api == "ollama" else "/v1/chat/completions")

        def driver() -> RequestResult:
            if api == "ollama":
                return _one_ollama(host, port, model, prompt, num_predict, timeout)
            return _one_openai(host, port, scheme, path, api_key, model, prompt, num_predict, timeout)

        results: list[RequestResult] = []
        done = 0
        lock = threading.Lock()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(driver) for _ in range(num_requests)]
            for f in as_completed(futures):
                r = f.result()
                with lock:
                    results.append(r)
                    done += 1
                if on_progress:
                    try:
                        on_progress(done, num_requests)
                    except Exception:
                        pass
        wall = time.perf_counter() - t0

        return _aggregate(
            endpoint=f"{scheme}://{host}:{port}{path}",
            model=model, api=api, concurrency=concurrency,
            num_requests=num_requests, wall_s=wall,
            per_request=results, slo=slo,
        )

    def sweep(
        self,
        endpoint: str,
        model: str,
        *,
        api: ApiFlavour = "ollama",
        concurrencies: list[int] = None,  # type: ignore[assignment]
        num_requests_per_step: int = 16,
        num_predict: int = 128,
        prompt: str = "Explain attention mechanisms in transformers.",
        api_key: Optional[str] = None,
        slo: Optional[SloTargets] = None,
        on_step: Optional[Callable[[LoadTestResult], None]] = None,
    ) -> SweepResult:
        concurrencies = concurrencies or [1, 2, 4, 8, 16]
        points: list[LoadTestResult] = []
        for c in concurrencies:
            res = self.run(
                endpoint=endpoint, model=model, prompt=prompt,
                api=api, concurrency=c, num_requests=max(num_requests_per_step, c),
                num_predict=num_predict, api_key=api_key, slo=slo,
            )
            points.append(res)
            if on_step:
                on_step(res)
        return SweepResult(
            endpoint=endpoint, model=model, points=points,
            knee_concurrency=_knee_of(points),
        )


def _knee_of(points: list[LoadTestResult]) -> Optional[int]:
    """Pick the concurrency after which aggregate tok/s grows <10 %.

    Classic throughput knee heuristic — saturated backend.
    """
    if len(points) < 2:
        return None
    sorted_pts = sorted(points, key=lambda p: p.concurrency)
    best = sorted_pts[0]
    for p in sorted_pts[1:]:
        gain = (p.aggregate_tok_s - best.aggregate_tok_s) / (best.aggregate_tok_s or 1)
        if gain < 0.10:
            return best.concurrency
        best = p
    return sorted_pts[-1].concurrency
