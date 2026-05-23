"""SIMPLE-EVO: a simplified successor to LEVI.

Keeps LEVI's frontier + mutation parallelism, error archive, self-repair, and
meta-advisor. Replaces CVT-MAP-Elites with description-based semantic niching,
the PPS stagnation formula with three sliding-window signals, and the 4D
bandit with two operators plus UCB-style selection.

See docs/SIMPLE_EVO.md for the full design.
"""

from .ast_signature import N_FEATURES, ast_cosine, compute_ast_signature
from .embedder import DescriptionEmbedder, EmbedderConfig
from .monitor import Monitor, MonitorConfig
from .parser import LLMOutput, OutputParser, ParserConfig
from .pool import Pool, PoolConfig, Program
from .selector import Selector, SelectorConfig

__all__ = [
    "DescriptionEmbedder",
    "EmbedderConfig",
    "LLMOutput",
    "Monitor",
    "MonitorConfig",
    "N_FEATURES",
    "OutputParser",
    "ParserConfig",
    "Pool",
    "PoolConfig",
    "Program",
    "Selector",
    "SelectorConfig",
    "ast_cosine",
    "compute_ast_signature",
]
