"""Formula-grounded explanations of plan outputs.

`MemoryWaterfall` decomposes a plan's memory budget into named terms, each
annotated with its formula and a citation URL. The goal: when someone argues
"your calculator says it won't fit but it runs", they can see every term and
decide which override makes sense.
"""

from kv_planner.core.explain.waterfall import MemoryWaterfall, WaterfallTerm

__all__ = ["MemoryWaterfall", "WaterfallTerm"]
