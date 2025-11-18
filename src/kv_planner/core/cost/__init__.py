"""
Cost analysis for LLM deployment.

Provides TCO (Total Cost of Ownership) analysis and cost modeling:
- GPU rental costs (cloud)
- $/million tokens
- Utilization curves
- Break-even analysis
"""

from kv_planner.core.cost.analyzer import CostAnalyzer, CostMetrics, GPUPricing

__all__ = [
    "CostAnalyzer",
    "CostMetrics",
    "GPUPricing",
]
