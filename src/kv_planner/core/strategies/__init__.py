"""
Optimization strategy analyzers.

Provides evaluation of different deployment strategies:
- Quantization (FP8, INT8, INT4)
- Prefix caching
- Disaggregation (prefill/decode separation)
"""

from kv_planner.core.strategies.quantization import (
    QuantizationEvaluator,
    QuantizationMetrics,
)
from kv_planner.core.strategies.caching import (
    PrefixCachingAnalyzer,
    PrefixCachingMetrics,
)

__all__ = [
    "QuantizationEvaluator",
    "QuantizationMetrics",
    "PrefixCachingAnalyzer",
    "PrefixCachingMetrics",
]
