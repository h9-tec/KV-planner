"""FastAPI dashboard — llmfit-parity REST surface on :8787.

Same routes as llmfit's ``llmfit serve`` so existing clients / schedulers
keep working, plus additional endpoints that exploit kv-planner's unique
physics (``/api/v1/fleet``, ``/api/v1/rationale``, ``/api/v1/training-plan``).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from kv_planner.application.fleet import FleetPlanner
from kv_planner.application.rationale import explain
from kv_planner.application.recommender import Recommender
from kv_planner.core.training import TrainingPlanner
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.hw_detect import detect
from kv_planner.infrastructure.model_catalog import CATALOG, by_slug
from kv_planner.infrastructure.runtime_probe import probe_all


def create_app() -> FastAPI:
    app = FastAPI(title="kv-planner", version="0.2.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/system")
    def api_system() -> dict:
        hw = detect()
        runtimes = probe_all()
        return {
            "cpu": {"model": hw.cpu_model, "cores": hw.cpu_cores},
            "ram": {
                "total_gb": round(hw.ram_total_gb, 1),
                "available_gb": round(hw.ram_available_gb, 1),
            },
            "gpu": {
                "vendor": hw.gpu_vendor,
                "name_raw": hw.gpu_name_raw,
                "vram_gb": round(hw.gpu_vram_gb, 2),
                "matched_db_key": hw.gpu_matched_db_key,
                "num_gpus": hw.num_gpus,
            },
            "runtimes": [
                {"name": r.name, "reachable": r.reachable,
                 "endpoint": r.endpoint, "models": r.models,
                 "version": r.version}
                for r in runtimes
            ],
        }

    # ---- Helpers ------------------------------------------------------
    def _gpu_or_detected(gpu: Optional[str]):
        if gpu:
            return GPUDatabase.to_hardware_spec(gpu)
        detected = detect().gpu_matched_db_key
        if not detected:
            raise HTTPException(400, "no GPU detected; pass ?gpu=<key>")
        return GPUDatabase.to_hardware_spec(detected)

    def _serialise_rec(r) -> dict:
        return {
            "slug": r.entry.slug,
            "name": r.entry.config.name,
            "provider": r.entry.provider,
            "precision": r.precision,
            "throughput_tok_s": round(r.throughput_tok_s, 1),
            "latency_ms": round(r.latency_ms, 1),
            "memory_gb": round(r.memory_gb, 2),
            "memory_util_pct": round(r.memory_util_pct, 1),
            "fits": r.fits,
            "score_quality": r.score_quality,
            "score_fit": r.score_fit,
            "score_speed": r.score_speed,
            "score_context": r.score_context,
            "score_composite": round(r.score_composite, 1),
            "license": r.entry.license,
            "ollama_tags": list(r.entry.ollama_tags),
            "use_cases": list(r.entry.use_cases),
        }

    # ---- /api/v1/models ----------------------------------------------
    @app.get("/api/v1/models")
    def api_models(
        gpu: Optional[str] = None,
        use_case: str = Query("general"),
        search: Optional[str] = None,
        provider: Optional[str] = None,
        min_fit: Optional[str] = Query(None, regex="^(perfect|good|marginal|too_tight|runnable)$"),
        include_unfit: bool = False,
        sort: str = Query("score", regex="^(score|params|mem|ctx|date|use_case)$"),
        limit: int = Query(50, ge=1, le=500),
        input_length: int = 2048,
        output_length: int = 512,
        batch_size: int = 1,
    ) -> dict:
        spec = _gpu_or_detected(gpu)
        recs = Recommender().recommend(
            spec, use_case=use_case,  # type: ignore[arg-type]
            input_length=input_length, output_length=output_length,
            batch_size=batch_size,
        )
        # filters
        if search:
            s = search.lower()
            recs = [r for r in recs if s in r.entry.slug.lower() or s in r.entry.provider.lower()]
        if provider:
            recs = [r for r in recs if r.entry.provider.lower() == provider.lower()]
        if min_fit == "runnable":
            recs = [r for r in recs if r.fits]
        elif min_fit:
            # map util % to fit tags
            def tag(r):
                if not r.fits or r.memory_util_pct > 100: return "too_tight"
                if r.memory_util_pct <= 50: return "perfect"
                if r.memory_util_pct <= 75: return "good"
                if r.memory_util_pct <= 95: return "marginal"
                return "too_tight"
            order = {"perfect": 0, "good": 1, "marginal": 2, "too_tight": 3}
            recs = [r for r in recs if order[tag(r)] <= order[min_fit]]
        if not include_unfit:
            recs = [r for r in recs if r.fits]
        # sort
        def key(r):
            if sort == "params":   return -r.entry.config.total_params()
            if sort == "mem":       return r.memory_util_pct
            if sort == "ctx":       return -r.entry.config.max_position_embeddings
            if sort == "date":      return r.entry.released or ""
            if sort == "use_case":  return r.entry.use_cases[0] if r.entry.use_cases else ""
            return -r.score_composite
        recs.sort(key=key)
        return {
            "gpu": gpu or detect().gpu_matched_db_key,
            "use_case": use_case,
            "count": len(recs[:limit]),
            "models": [_serialise_rec(r) for r in recs[:limit]],
        }

    @app.get("/api/v1/models/top")
    def api_top(
        gpu: Optional[str] = None,
        use_case: str = Query("general"),
        limit: int = Query(5, ge=1, le=50),
    ) -> dict:
        spec = _gpu_or_detected(gpu)
        recs = Recommender().top_n(
            spec, n=limit, use_case=use_case,  # type: ignore[arg-type]
        )
        return {
            "gpu": gpu or detect().gpu_matched_db_key,
            "use_case": use_case,
            "models": [_serialise_rec(r) for r in recs],
        }

    @app.get("/api/v1/models/{slug}")
    def api_model(slug: str, gpu: Optional[str] = None) -> dict:
        entry = by_slug(slug)
        if not entry:
            raise HTTPException(404, f"unknown slug {slug}")
        spec = _gpu_or_detected(gpu)
        rec = Recommender().recommend(spec, use_case=entry.use_cases[0] if entry.use_cases else "general")
        matching = [r for r in rec if r.entry.slug == slug]
        if not matching:
            raise HTTPException(500, "recommender returned no matching row")
        r = matching[0]
        rat = explain(r, spec)
        return {
            **_serialise_rec(r),
            "rationale": {"verdict": rat.verdict, "bullets": rat.bullets, "caveats": rat.caveats},
        }

    @app.get("/api/v1/fleet")
    def api_fleet(
        model: str,
        rps: float = Query(..., gt=0),
        p99_latency_ms: float = Query(2000.0, gt=0),
        input_length: int = 2048,
        output_length: int = 512,
        gpus: Optional[str] = None,
        tp: Optional[str] = None,
        precisions: str = "fp16,fp8,int4",
        limit: int = Query(10, ge=1, le=100),
    ) -> dict:
        entry = by_slug(model)
        if not entry:
            raise HTTPException(404, f"unknown model slug {model}")
        gpu_list = (gpus or "H100-SXM-80GB,A100-SXM-80GB,RTX-5090").split(",")
        tp_list = tuple(int(x) for x in (tp or "1,2,4,8").split(","))
        prec_list = tuple(p.strip() for p in precisions.split(","))

        designs = FleetPlanner().design(
            entry.config, target_rps=rps,
            input_length=input_length, output_length=output_length,
            gpu_candidates=[g.strip() for g in gpu_list],
            p99_latency_ms=p99_latency_ms,
            precisions=prec_list,  # type: ignore[arg-type]
            tp_candidates=tp_list,
        )
        return {
            "model": model,
            "designs": [
                {
                    "gpu": d.gpu_model, "tp": d.tp_size, "replicas": d.replicas,
                    "total_gpus": d.total_gpus, "precision": d.precision,
                    "per_replica_tok_s": round(d.per_replica_throughput_tok_s, 0),
                    "aggregate_tok_s": round(d.aggregate_throughput_tok_s, 0),
                    "p99_latency_ms": round(d.p99_latency_ms, 0),
                    "meets_slo": d.meets_slo,
                    "cost_per_hour": round(d.cost_per_hour, 2),
                    "cost_per_million_tokens": round(d.cost_per_million_tokens, 3),
                }
                for d in designs[:limit]
            ],
        }

    @app.get("/api/v1/training-plan")
    def api_training_plan(
        model: str,
        gpu: Optional[str] = None,
        method: str = "qlora",
        precision: str = "bf16",
        batch_size: int = 2,
        sequence_length: int = 2048,
        num_epochs: float = 1.0,
        dataset_tokens: int = 1_000_000,
        lora_rank: int = 16,
    ) -> dict:
        entry = by_slug(model)
        if not entry:
            raise HTTPException(404, f"unknown model slug {model}")
        spec = _gpu_or_detected(gpu)
        plan = TrainingPlanner().plan(
            entry.config, spec,
            method=method,  # type: ignore[arg-type]
            precision=precision,  # type: ignore[arg-type]
            batch_size=batch_size, sequence_length=sequence_length,
            num_epochs=int(num_epochs) if num_epochs >= 1 else 1,
            dataset_tokens=dataset_tokens, lora_rank=lora_rank,
        )
        return {
            "model": model, "gpu": spec.gpu_model, "method": plan.method,
            "memory": {
                "weights_gb": round(plan.model_weight_gb, 2),
                "gradients_gb": round(plan.gradient_gb, 2),
                "optimizer_gb": round(plan.optimizer_state_gb, 2),
                "activations_gb": round(plan.activation_gb, 2),
                "total_gb": round(plan.total_memory_gb, 2),
                "fits": plan.fits_per_gpu,
            },
            "compute": {
                "tokens_per_second": round(plan.tokens_per_second, 1),
                "estimated_hours": round(plan.est_training_hours, 2),
                "estimated_cost_usd": round(plan.est_cost_usd, 2),
            },
            "trainable_params_m": round(plan.trainable_params / 1e6, 2),
        }

    # ---- Minimal HTML landing page ------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        hw = detect()
        gpu = hw.gpu_matched_db_key or "no GPU"
        return (
            f"<!doctype html><html><head>"
            f"<title>kv-planner</title>"
            f"<style>"
            f"body{{font-family:ui-monospace,Menlo,monospace;background:#191b22;color:#e5e5e9;"
            f"padding:2rem;max-width:920px;margin:auto}}"
            f"h1{{color:#fa9450}} h2{{color:#4abfad;margin-top:1.5rem}}"
            f"a{{color:#fa9450}} code{{color:#a2c761}}"
            f"</style>"
            f"</head><body>"
            f"<h1>kv-planner</h1>"
            f"<p>llmfit-style physics-scored recommender. Detected: "
            f"<code>{gpu}</code>, {hw.gpu_vram_gb:.0f} GB VRAM.</p>"
            f"<h2>JSON endpoints</h2>"
            f"<ul>"
            f"<li><a href='/api/v1/system'>/api/v1/system</a></li>"
            f"<li><a href='/api/v1/models/top?limit=5'>/api/v1/models/top?limit=5</a></li>"
            f"<li><a href='/api/v1/models?use_case=coding&limit=10'>/api/v1/models?use_case=coding</a></li>"
            f"<li><a href='/api/v1/models/llama-3-8b'>/api/v1/models/llama-3-8b</a></li>"
            f"<li><a href='/api/v1/fleet?model=llama-3-8b&rps=20'>/api/v1/fleet?model=llama-3-8b&rps=20</a></li>"
            f"<li><a href='/api/v1/training-plan?model=llama-3-8b'>/api/v1/training-plan?model=llama-3-8b</a></li>"
            f"</ul>"
            f"<p><a href='/docs'>interactive OpenAPI docs</a></p>"
            f"</body></html>"
        )

    return app


def run_background(host: str, port: int) -> str:
    """Start uvicorn in a daemon thread; return the URL to connect to."""
    import threading

    import uvicorn

    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="kvp-dashboard")
    thread.start()
    return f"http://{host}:{port}"


def run_foreground(host: str, port: int) -> None:
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
