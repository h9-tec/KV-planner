"""
Export utilities for deployment plans.

Supports JSON, YAML, and formatted text output.
"""

import json
from typing import Any

from kv_planner.application.planner import DeploymentPlan


def to_dict(plan: DeploymentPlan) -> dict[str, Any]:
    """
    Convert DeploymentPlan to dictionary.

    Args:
        plan: Deployment plan

    Returns:
        Dictionary representation
    """
    return {
        "model": {
            "name": plan.model.name,
            "num_layers": plan.model.num_layers,
            "hidden_size": plan.model.hidden_size,
            "num_attention_heads": plan.model.num_attention_heads,
            "num_key_value_heads": plan.model.num_key_value_heads,
            "vocab_size": plan.model.vocab_size,
            "max_position_embeddings": plan.model.max_position_embeddings,
        },
        "hardware": {
            "gpu_model": plan.hardware.gpu_model,
            "num_gpus": plan.hardware.num_gpus,
            "gpu_memory_gb": plan.hardware.gpu_memory_gb,
            "peak_tflops": plan.hardware.peak_tflops,
            "hbm_bandwidth_gb_s": plan.hardware.hbm_bandwidth_gb_s,
        },
        "recommendations": {
            "precision": plan.recommended_precision,
            "batch_size": plan.recommended_batch_size,
            "enable_prefix_caching": plan.enable_prefix_caching,
            "prefix_cache_length": plan.prefix_cache_length,
            "optimization_goal": plan.optimization_goal,
        },
        "performance": {
            "throughput_tokens_per_sec": plan.performance.throughput_tokens_per_sec,
            "prefill_latency_ms": plan.performance.prefill_latency_ms,
            "decode_latency_ms": plan.performance.decode_latency_ms,
            "total_latency_ms": plan.performance.total_latency_ms,
            "mfu": plan.performance.mfu,
            "mbu": plan.performance.mbu,
        },
        "cost": {
            "cost_per_hour": plan.cost.cost_per_hour,
            "cost_per_million_tokens": plan.cost.cost_per_million_tokens,
            "cost_per_request": plan.cost.cost_per_request,
            "monthly_cost_usd": plan.cost.monthly_cost_usd,
            "annual_cost_usd": plan.cost.annual_cost_usd,
            "utilization_pct": plan.cost.utilization_pct,
            "tokens_per_dollar": plan.cost.tokens_per_dollar,
        },
        "memory": {
            "model_memory_gb": plan.model_memory_gb,
            "kv_cache_memory_gb": plan.kv_cache_memory_gb,
            "total_memory_gb": plan.total_memory_gb,
            "memory_utilization_pct": plan.memory_utilization_pct,
        },
        "quantization": {
            "precision": plan.quantization.precision,
            "method": plan.quantization.method,
            "memory_savings_pct": plan.quantization.memory_savings_pct,
            "speed_improvement": plan.quantization.speed_improvement,
            "perplexity_delta": plan.quantization.perplexity_delta,
            "quality_impact": plan.quantization.quality_impact,
        },
        "caching": {
            "enabled": plan.enable_prefix_caching,
            "prefix_length": plan.prefix_cache_length,
            "memory_savings_pct": plan.caching.memory_savings_pct if plan.caching else 0.0,
            "latency_reduction_pct": plan.caching.latency_reduction_pct if plan.caching else 0.0,
            "cost_savings_pct": plan.caching.cost_savings_pct if plan.caching else 0.0,
        } if plan.caching else None,
        "vllm_config": plan.vllm_config,
    }


def to_json(plan: DeploymentPlan, indent: int = 2) -> str:
    """
    Export plan to JSON.

    Args:
        plan: Deployment plan
        indent: JSON indentation

    Returns:
        JSON string
    """
    return json.dumps(to_dict(plan), indent=indent)


def to_yaml(plan: DeploymentPlan) -> str:
    """
    Export plan to YAML.

    Args:
        plan: Deployment plan

    Returns:
        YAML string
    """
    try:
        import yaml
        return yaml.dump(to_dict(plan), default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to simple format if yaml not available
        return _to_simple_yaml(to_dict(plan))


def _to_simple_yaml(data: dict, indent: int = 0) -> str:
    """Simple YAML-like formatter (fallback)."""
    lines = []
    prefix = "  " * indent

    for key, value in data.items():
        if value is None:
            lines.append(f"{prefix}{key}: null")
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_to_simple_yaml(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.append(_to_simple_yaml(item, indent + 2))
                else:
                    lines.append(f"{prefix}  - {item}")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {str(value).lower()}")
        elif isinstance(value, str):
            lines.append(f"{prefix}{key}: {value}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


def to_markdown(plan: DeploymentPlan) -> str:
    """
    Export plan to Markdown report.

    Args:
        plan: Deployment plan

    Returns:
        Markdown string
    """
    lines = [
        "# Deployment Plan Report",
        "",
        f"**Model**: {plan.model.name}",
        f"**Hardware**: {plan.hardware.num_gpus}× {plan.hardware.gpu_model}",
        f"**Optimization Goal**: {plan.optimization_goal}",
        "",
        "## Recommendations",
        "",
        f"- **Precision**: {plan.recommended_precision.upper()} ({plan.quantization.quality_impact} quality impact)",
        f"- **Batch Size**: {plan.recommended_batch_size}",
        f"- **Prefix Caching**: {'Enabled' if plan.enable_prefix_caching else 'Disabled'}",
        "",
        "## Performance",
        "",
        f"- **Throughput**: {plan.performance.throughput_tokens_per_sec:,.0f} tokens/sec",
        f"- **Latency**: {plan.performance.total_latency_ms:.0f} ms",
        f"  - Prefill: {plan.performance.prefill_latency_ms:.0f} ms",
        f"  - Decode: {plan.performance.decode_latency_ms:.0f} ms",
        f"- **MFU**: {plan.performance.mfu*100:.1f}%",
        f"- **MBU**: {plan.performance.mbu*100:.1f}%",
        "",
        "## Cost",
        "",
        f"- **Cost per hour**: ${plan.cost.cost_per_hour:.2f}",
        f"- **Cost per million tokens**: ${plan.cost.cost_per_million_tokens:.2f}",
        f"- **Monthly cost**: ${plan.cost.monthly_cost_usd:,.2f}",
        f"- **GPU utilization**: {plan.cost.utilization_pct:.1f}%",
        "",
        "## Memory",
        "",
        f"- **Model**: {plan.model_memory_gb:.2f} GB",
        f"- **KV Cache**: {plan.kv_cache_memory_gb:.2f} GB",
        f"- **Total**: {plan.total_memory_gb:.2f} GB ({plan.memory_utilization_pct:.1f}% of {plan.hardware.gpu_memory_gb:.0f} GB)",
        "",
        "## Savings",
        "",
        f"- **Memory savings**: {plan.quantization.memory_savings_pct:.0f}% (vs FP16)",
        f"- **Speed improvement**: {plan.quantization.speed_improvement:.2f}× (vs FP16)",
    ]

    if plan.caching:
        lines.extend([
            f"- **Caching latency reduction**: {plan.caching.latency_reduction_pct:.0f}%",
            f"- **Caching memory savings**: {plan.caching.memory_savings_pct:.0f}%",
        ])

    lines.extend([
        "",
        "## vLLM Configuration",
        "",
        "```python",
        json.dumps(plan.vllm_config, indent=2),
        "```",
        "",
    ])

    return "\n".join(lines)


def save(plan: DeploymentPlan, filepath: str, format: str = "auto") -> None:
    """
    Save plan to file.

    Args:
        plan: Deployment plan
        filepath: Output file path
        format: Format ('json', 'yaml', 'md', or 'auto' to detect from extension)
    """
    if format == "auto":
        if filepath.endswith(".json"):
            format = "json"
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            format = "yaml"
        elif filepath.endswith(".md"):
            format = "md"
        else:
            format = "json"  # Default

    if format == "json":
        content = to_json(plan)
    elif format == "yaml":
        content = to_yaml(plan)
    elif format == "md":
        content = to_markdown(plan)
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(filepath, "w") as f:
        f.write(content)
