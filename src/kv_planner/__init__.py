"""
kv-planner: KV cache memory and throughput planner for LLM deployment.

Public API (stable at the package root):

* :class:`ModelConfig`, :class:`HardwareSpec`, :class:`DeploymentConstraints`,
  :class:`TrafficModel`, :class:`Distribution`  — domain value objects.
* :class:`DeploymentPlanner`, :class:`DeploymentPlan` — orchestration.
* :mod:`kv_planner.export` — JSON / YAML / Markdown serializers.
* :data:`__version__`.
"""

from kv_planner.application import DeploymentPlan, DeploymentPlanner, OptimizationGoal, export
from kv_planner.domain import (
    DeploymentConstraints,
    Distribution,
    HardwareSpec,
    InsufficientMemoryError,
    InvalidConfigurationError,
    KVPlannerError,
    ModelConfig,
    PrecisionType,
    TrafficModel,
)
from kv_planner.version import __version__

__all__ = [
    "DeploymentConstraints",
    "DeploymentPlan",
    "DeploymentPlanner",
    "Distribution",
    "HardwareSpec",
    "InsufficientMemoryError",
    "InvalidConfigurationError",
    "KVPlannerError",
    "ModelConfig",
    "OptimizationGoal",
    "PrecisionType",
    "TrafficModel",
    "__version__",
    "export",
]
