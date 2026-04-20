"""Single source of truth for numerical-precision byte sizes.

Every formula in this package that converts between tensor elements and bytes
should read its coefficients from here. Replicating this table in multiple
modules has historically caused the same bug to be fixed in one place and
regress in another.

Values are bytes per element. ``int4`` is 0.5 in the theoretical storage
sense (4 bits). Real kernels sometimes pack 2 int4s per byte and sometimes
pad to int8 for alignment — :func:`bytes_per_element` returns the theoretical
storage, and callers that care about kernel layout should annotate explicitly.
"""

from typing import Literal, Mapping

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


_BYTES_PER_ELEMENT: Mapping[PrecisionType, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


def bytes_per_element(precision: PrecisionType) -> float:
    """Return bytes per tensor element for ``precision``."""
    try:
        return _BYTES_PER_ELEMENT[precision]
    except KeyError as exc:
        raise ValueError(
            f"Unknown precision {precision!r}; expected one of {list(_BYTES_PER_ELEMENT)}"
        ) from exc


def supported_precisions() -> tuple[PrecisionType, ...]:
    """Tuple of every precision this module knows about."""
    return tuple(_BYTES_PER_ELEMENT.keys())
