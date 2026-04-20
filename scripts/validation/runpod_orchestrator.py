"""RunPod validation orchestrator.

Uses RunPod's GraphQL API to spin up pods for each (gpu, model, precision)
configuration, executes ``in_pod_validate.sh`` over SSH, pulls back the
result JSON, and tears the pod down. Has a hard cost cap so a runaway
run can't spend more than the budget.

Usage::

    export RUNPOD_API_KEY=...
    export HF_TOKEN=...              # for gated models (Llama family)
    python scripts/validation/runpod_orchestrator.py --budget-usd 25

Without ``--budget-usd`` the script prints the matrix + estimated cost and
exits without provisioning anything. ``--dry-run`` is implicit on first
invocation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import urllib.request
    import urllib.error
except ImportError:
    sys.exit("Python 3 stdlib required")


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    gpu_runpod_id: str       # RunPod GPU type ID (see runpod.io/console/gpus)
    gpu_db_key: str          # kv-planner's key in gpu_specs.py
    model_hf: str            # HF model id
    model_slug: str          # kv-planner catalog slug
    precision: str = "fp16"
    input_length: int = 2048
    output_length: int = 256
    concurrency: int = 8
    num_requests: int = 32
    estimated_pod_minutes: int = 18    # setup + download + test + teardown
    estimated_hourly_usd: float = 2.50  # rough on-demand across regions

    @property
    def estimated_cost_usd(self) -> float:
        return self.estimated_pod_minutes / 60.0 * self.estimated_hourly_usd


# Minimal matrix — dense on enterprise + prosumer. MoE validation needs
# larger VRAM so it's a separate expanded run.
DEFAULT_MATRIX: list[Config] = [
    # H100 SXM 80GB — the production reference
    Config("NVIDIA H100 80GB HBM3", "H100-SXM-80GB",
           "meta-llama/Meta-Llama-3-8B-Instruct", "llama-3-8b",
           precision="fp16", estimated_hourly_usd=3.99),
    Config("NVIDIA H100 80GB HBM3", "H100-SXM-80GB",
           "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b",
           precision="fp16", estimated_hourly_usd=3.99),

    # A100 80GB — second most common production GPU
    Config("NVIDIA A100 80GB PCIe", "A100-PCIe-80GB",
           "meta-llama/Meta-Llama-3-8B-Instruct", "llama-3-8b",
           precision="fp16", estimated_hourly_usd=1.89),

    # L40S — cost-optimised serving tier
    Config("NVIDIA L40S", "L40S",
           "meta-llama/Meta-Llama-3-8B-Instruct", "llama-3-8b",
           precision="fp16", estimated_hourly_usd=1.19),

    # RTX 4090 — prosumer validation
    Config("NVIDIA GeForce RTX 4090", "RTX-4090",
           "meta-llama/Meta-Llama-3-8B-Instruct", "llama-3-8b",
           precision="fp16", estimated_hourly_usd=0.44),
]


# Extended MoE matrix — enable with --include-moe. 70B and Mixtral need
# more VRAM or multi-GPU; skip laptops.
MOE_MATRIX: list[Config] = [
    Config("NVIDIA H100 80GB HBM3", "H100-SXM-80GB",
           "mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral-8x7b",
           precision="fp16", estimated_pod_minutes=30, estimated_hourly_usd=3.99),
]


# ---------------------------------------------------------------------------
# RunPod GraphQL
# ---------------------------------------------------------------------------


RUNPOD_ENDPOINT = "https://api.runpod.io/graphql"


def _gql(query: str, variables: dict, api_key: str) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        RUNPOD_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # RunPod's Cloudflare WAF blocks the default Python UA. Use a
            # generic curl-like UA to get past the integrity check.
            "User-Agent": "kv-planner-validator/0.3 (+https://github.com/h9-tec/KV-planner)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod HTTP {e.code}: {e.read().decode()[:500]}") from e
    if data.get("errors"):
        raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
    return data["data"]


def list_available_gpus(api_key: str) -> list[dict]:
    query = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        lowestPrice(input:{gpuCount:1}) { minimumBidPrice uninterruptablePrice }
      }
    }
    """
    return _gql(query, {}, api_key).get("gpuTypes", [])


def get_my_pubkey(api_key: str) -> str:
    """Fetch the user's registered SSH public key — needed so the pod's
    /start.sh actually launches sshd."""
    data = _gql("{ myself { pubKey } }", {}, api_key)
    return ((data.get("myself") or {}).get("pubKey") or "").strip()


def create_pod(
    api_key: str,
    gpu_type: str,
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    cloud_type: str = "ALL",
    pubkey: str | None = None,
) -> str:
    """Create a pod; try COMMUNITY first, then SECURE if supply-constrained.

    ``cloud_type="ALL"`` means: try COMMUNITY, and if it fails with a
    SUPPLY_CONSTRAINT error, fall back to SECURE automatically.
    """
    query = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        desiredStatus
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
      }
    }
    """
    cloud_types_to_try = (
        ["COMMUNITY", "SECURE"] if cloud_type == "ALL" else [cloud_type]
    )
    env_vars = []
    if pubkey:
        # runpod/pytorch images launch sshd automatically when PUBLIC_KEY is set.
        env_vars.append({"key": "PUBLIC_KEY", "value": pubkey})
    last_err: Exception | None = None
    for ct in cloud_types_to_try:
        variables = {"input": {
            "cloudType": ct,
            "gpuCount": 1,
            "volumeInGb": 30,
            "containerDiskInGb": 20,
            "minVcpuCount": 2,
            "minMemoryInGb": 16,
            "gpuTypeId": gpu_type,
            "name": f"kvp-val-{int(time.time())}",
            "imageName": image,
            "ports": "22/tcp,8000/http",
            "volumeMountPath": "/workspace",
            "env": env_vars,
        }}
        try:
            pod = _gql(query, variables, api_key)["podFindAndDeployOnDemand"]
            if pod and pod.get("id"):
                return pod["id"]
        except RuntimeError as e:
            last_err = e
            # Only fall through to next cloud type on supply / resource errors.
            if "SUPPLY_CONSTRAINT" in str(e) or "resources to deploy" in str(e):
                continue
            raise
    raise last_err or RuntimeError(f"create_pod failed for {gpu_type}")


def get_pod(api_key: str, pod_id: str) -> dict:
    query = """
    query GetPod($podId: String!) {
      pod(input: {podId: $podId}) {
        id
        desiredStatus
        runtime {
          uptimeInSeconds
          ports { ip isIpPublic privatePort publicPort type }
        }
      }
    }
    """
    return _gql(query, {"podId": pod_id}, api_key)["pod"]


def terminate_pod(api_key: str, pod_id: str) -> None:
    query = "mutation { podTerminate(input: {podId: \"%s\"}) }" % pod_id
    _gql(query, {}, api_key)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    config: Config
    status: str   # "success" | "failed" | "timeout"
    result_json: Optional[dict] = None
    error: str = ""
    wall_minutes: float = 0.0


def print_matrix_summary(matrix: list[Config]) -> None:
    print()
    print("Configured test matrix:")
    print(f"  {'GPU':<24} {'Model':<36} {'Prec':<6} {'Est. min':>8} {'Est. $':>7}")
    print("  " + "-" * 86)
    total = 0.0
    for c in matrix:
        print(f"  {c.gpu_db_key:<24} {c.model_slug:<36} {c.precision:<6} "
              f"{c.estimated_pod_minutes:>8} ${c.estimated_cost_usd:>6.2f}")
        total += c.estimated_cost_usd
    print("  " + "-" * 86)
    print(f"  Total estimated cost: ${total:.2f}")
    print()


def run_one(
    cfg: Config, api_key: str, hf_token: str,
    out_dir: pathlib.Path, ssh_opts: str = "",
) -> RunResult:
    print(f"\n[{cfg.gpu_db_key} × {cfg.model_slug}]  creating pod…")
    t0 = time.time()
    pod_id: Optional[str] = None
    try:
        pubkey = get_my_pubkey(api_key)
        if not pubkey:
            return RunResult(cfg, "failed",
                error="no SSH key registered — add ~/.ssh/id_ed25519.pub at runpod.io/console/user/settings")
        pod_id = create_pod(api_key, cfg.gpu_runpod_id, pubkey=pubkey)
        print(f"  pod_id={pod_id}, waiting for RUNNING + SSH port…")
        # Wait up to 15 min for RUNNING + port (heavy images pull slowly)
        pod = None
        for i in range(90):
            time.sleep(10)
            pod = get_pod(api_key, pod_id)
            status = pod.get("desiredStatus") if pod else "?"
            runtime = (pod or {}).get("runtime") or {}
            ports = runtime.get("ports") or []
            if status == "RUNNING" and ports:
                print(f"  pod RUNNING after {(i+1)*10}s")
                break
            if i % 6 == 5:
                print(f"  [{(i+1)*10}s] status={status} ports={len(ports)}")
        else:
            terminate_pod(api_key, pod_id)
            return RunResult(cfg, "timeout", error=f"pod never reached RUNNING (last status={status})")

        ssh_port = None
        ssh_host = None
        runtime = (pod or {}).get("runtime") or {}
        for p in (runtime.get("ports") or []):
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                ssh_port = p.get("publicPort")
                ssh_host = p.get("ip")
        if ssh_port is None:
            terminate_pod(api_key, pod_id)
            return RunResult(cfg, "failed", error="no public SSH port")

        print(f"  pod running: {ssh_host}:{ssh_port} — waiting for sshd")

        # sshd inside the container usually needs another 30–90 s after
        # the pod reaches RUNNING to accept connections.
        import subprocess
        ssh_ready = False
        for i in range(12):   # up to 2 minutes
            probe = subprocess.run(
                f"ssh -p {ssh_port} {ssh_opts} "
                f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
                f"-o ConnectTimeout=5 -o BatchMode=yes "
                f"root@{ssh_host} 'echo ready'",
                shell=True, capture_output=True, timeout=15,
            )
            if probe.returncode == 0 and b"ready" in probe.stdout:
                ssh_ready = True
                print(f"  sshd ready after {(i+1)*10}s")
                break
            time.sleep(10)
        if not ssh_ready:
            return RunResult(cfg, "failed", error="sshd never answered on assigned port")

        # SSH + execute harness
        remote_cmd = (
            f"cd /workspace && "
            f"git clone https://github.com/h9-tec/KV-planner.git repo && "
            f"cd repo && "
            f"export HF_TOKEN='{hf_token}' && "
            f"bash scripts/validation/in_pod_validate.sh "
            f"'{cfg.model_hf}' '{cfg.gpu_db_key}' '{cfg.model_slug}' '{cfg.precision}'"
        )
        cmd = (
            f"ssh -p {ssh_port} {ssh_opts} "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"root@{ssh_host} \"{remote_cmd}\""
        )
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=cfg.estimated_pod_minutes * 60 + 300,
        )

        # Pull back the JSON
        scp_cmd = (
            f"scp -P {ssh_port} {ssh_opts} "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"root@{ssh_host}:/workspace/validation_result.json "
            f"{out_dir}/{cfg.gpu_db_key}_{cfg.model_slug}.json"
        )
        subprocess.run(scp_cmd, shell=True, timeout=60)

        path = out_dir / f"{cfg.gpu_db_key}_{cfg.model_slug}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            return RunResult(cfg, "success", payload, "", (time.time() - t0) / 60)
        return RunResult(cfg, "failed", error="no result JSON produced")

    except Exception as e:
        return RunResult(cfg, "failed", error=str(e)[:300])
    finally:
        if pod_id:
            try:
                terminate_pod(api_key, pod_id)
                print(f"  pod {pod_id} terminated")
            except Exception:
                print(f"  WARNING: pod {pod_id} may still be running — terminate manually!")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="Hard cap on total spend. Without this, print matrix and exit.")
    ap.add_argument("--include-moe", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke-test: run only the cheapest config (RTX-4090) to validate the pipeline")
    ap.add_argument("--ssh-key", default="~/.ssh/id_ed25519",
                    help="SSH key path to give RunPod (must be added in console).")
    ap.add_argument("--out-dir", default="docs/validation_results")
    args = ap.parse_args()

    matrix = list(DEFAULT_MATRIX)
    if args.include_moe:
        matrix += MOE_MATRIX
    if args.smoke:
        # Just the cheapest config — smallest supply-constraint risk too.
        matrix = [c for c in matrix if c.gpu_db_key == "RTX-4090"] or matrix[:1]
    print_matrix_summary(matrix)

    total_cost = sum(c.estimated_cost_usd for c in matrix)
    if args.budget_usd is None:
        print("Pass --budget-usd <N> to actually launch. Set N >= "
              f"${total_cost:.2f} to cover the matrix above.")
        return 0

    if args.budget_usd < total_cost:
        print(f"ABORT: budget ${args.budget_usd:.2f} < estimated cost ${total_cost:.2f}.")
        return 2

    api_key = os.environ.get("RUNPOD_API_KEY")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not api_key:
        print("RUNPOD_API_KEY env var missing.", file=sys.stderr)
        return 1
    if not hf_token:
        print("Warning: HF_TOKEN not set — gated models (Llama family) will fail to download.",
              file=sys.stderr)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    spent = 0.0
    for cfg in matrix:
        if spent + cfg.estimated_cost_usd > args.budget_usd:
            print(f"Budget exhausted after ${spent:.2f}. Stopping.")
            break
        r = run_one(cfg, api_key, hf_token, out_dir,
                    ssh_opts=f"-i {os.path.expanduser(args.ssh_key)}")
        results.append(r)
        spent += cfg.estimated_cost_usd
        print(f"  → {r.status}{(' · ' + r.error) if r.error else ''}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION CAMPAIGN COMPLETE")
    print("=" * 80)
    print(f"Configs attempted:  {len(results)}")
    print(f"Succeeded:          {sum(1 for r in results if r.status == 'success')}")
    print(f"Estimated spent:    ${spent:.2f}")
    print()
    for r in results:
        mark = "✓" if r.status == "success" else "✗"
        line = f"  {mark} {r.config.gpu_db_key:<18} {r.config.model_slug:<24}  {r.status}"
        if r.status == "success" and r.result_json:
            mape = r.result_json.get("accuracy", {}).get("mape_tpot_pct")
            line += f"   MAPE(TPOT) = {mape}%" if mape is not None else ""
        print(line)
    print(f"\nPer-config JSON artifacts: {out_dir}/")
    return 0 if all(r.status == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
