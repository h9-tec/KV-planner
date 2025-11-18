"""Domain-specific exceptions."""


class KVPlannerError(Exception):
    """Base exception for kv-planner."""

    pass


class InvalidConfigurationError(KVPlannerError):
    """Raised when configuration is invalid or violates invariants."""

    pass


class InsufficientMemoryError(KVPlannerError):
    """Raised when GPU memory is insufficient for the requested configuration."""

    pass


class BenchmarkValidationError(KVPlannerError):
    """Raised when benchmark validation fails."""

    pass


class ModelLoadError(KVPlannerError):
    """Raised when model loading fails."""

    pass
