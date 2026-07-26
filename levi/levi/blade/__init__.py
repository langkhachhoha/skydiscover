"""BLADE — Behavior-Latent Adaptive Discovery Engine.

BLADE is the production wiring of SIMPLE-EVO. It reuses LEVI's frontier
prompts (3-phase paradigm shift), error-archive self-repair, and async
producer/consumer parallelism while swapping three subsystems:

* The behavioral archive becomes a top-K description-embedding pool
  (``levi.simple.Pool``).
* The stagnation signal becomes three sliding-window stats
  (``levi.simple.Monitor``).
* The 4-D Thompson bandit becomes a UCB-style selector
  (``levi.simple.Selector``).

The entry point is :func:`levi.methods.blade.evolve_code_blade`, which
matches :func:`levi.evolve_code` so existing problem definitions and
benchmarks plug in unchanged.
"""

from .orchestrator import (
    BladeConfig,
    BladeOrchestrator,
    BladeResult,
    ParadigmTrial,
    format_error_rate_table,
)
from .snaplog import SnapLog, classify_producer

__all__ = [
    "BladeConfig",
    "BladeOrchestrator",
    "BladeResult",
    "ParadigmTrial",
    "SnapLog",
    "classify_producer",
    "format_error_rate_table",
]
