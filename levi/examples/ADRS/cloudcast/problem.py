"""
Cloudcast Broadcast Optimization problem definition.

Contains prompts, seed program, and scoring function for `levi.evolve_code`.
"""

import collections
import heapq
import importlib
import json
import math
import os
import random
import sys
import tempfile
import time as time_module
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    import networkx as nx
except ModuleNotFoundError as e:
    if e.name == "networkx":
        raise RuntimeError(
            "Cloudcast requires networkx. From the repo root, run:\n"
            "  uv sync --extra example-cloudcast"
        ) from e
    raise
import numpy as np

PROBLEM_DESCRIPTION = """
# Cloudcast Broadcast Optimization

## Problem
Optimize broadcast topology for multi-cloud data distribution. Find optimal paths from a
source to multiple destinations across AWS, Azure, and GCP to minimize total transfer COST.
(Transfer time is reported for information only — the score depends solely on cost.)

## Key Concepts
- Graph G has edge attributes: `cost` ($/GB) and `throughput` (Gbps)
- BroadCastTopology stores paths for each (destination, partition) pair
- Data is partitioned into `num_partitions` chunks that can take different paths
- Paths from source must cover all destinations for all partitions

## Objective
Minimize total transfer cost ($/GB) across 5 network configurations:
- intra_aws, intra_azure, intra_gcp (single cloud)
- inter_agz, inter_gaz2 (cross-cloud)

## BroadCastTopology — USE THE PROVIDED CLASS, DO NOT REDEFINE IT
`BroadCastTopology` is already available in scope (do NOT write your own
class — redefining it will break evaluation). Construct and fill it via:
- `bc = BroadCastTopology(src, dsts, num_partitions)` - create topology
- `bc.append_dst_partition_path(dst, partition, [src, tgt, edge_data])` - add a
  path segment for one (destination, partition). `edge_data` must be `G[src][tgt]`.

The object you RETURN is consumed by an external simulator that reads these
attributes/methods directly, so your returned object MUST support all of them:
- attribute `bc.src` (str), `bc.dsts` (list[str]), `bc.num_partitions` (int)
- attribute `bc.paths`: dict[dst][str(partition)] -> list of [src, tgt, edge_data]
- method `bc.set_num_partitions(n)`
- method `bc.append_dst_partition_path(dst, partition, segment)`
The simplest correct approach is to use the provided `BroadCastTopology`
unchanged and only evolve the path-finding logic inside `search_algorithm`.

## Scoring (0-100)
```
LOWER_COST = 1199.00  # worst case
UPPER_COST = 626.24   # best known
cost_clamped = max(min(total_cost, LOWER_COST), UPPER_COST)
normalized_cost = (LOWER_COST - cost_clamped) / (LOWER_COST - UPPER_COST)
score = normalized_cost * 100
```

## APIs
- `G.nodes` - All nodes (cloud regions)
- `G.edges(data=True)` - All edges with attributes
- `G[src][dst]['cost']` - Cost per GB for edge
- `G[src][dst]['throughput']` - Throughput in Gbps
- `nx.dijkstra_path(G, src, dst, weight='cost')` - Shortest path by cost
- `BroadCastTopology(src, dsts, num_partitions)` - Create topology
- `bc_topology.append_dst_partition_path(dst, partition, [src, tgt, edge_data])` - Add path segment

## CRITICAL CONSTRAINTS (the evaluator REJECTS any topology that violates these)
- Cover EVERY (destination, partition) pair: each must have a non-empty path.
  Missing or empty partitions => the whole candidate is rejected (score 0).
- Each path must be a CONTINUOUS chain of edges from `src` to that destination:
  the segments for one (dst, partition) must connect end-to-end starting at `src`
  and ending at `dst` (segment k's target == segment k+1's source). A path that
  starts mid-graph or has a gap is rejected as "path discontinuity".
- Every segment must be a real edge of G (use `G[s][t]` as its `edge_data`).
- Do not add, drop, or duplicate destinations; keep `bc.src == src`.
- RETURN the filled `BroadCastTopology` object itself — not a dict, not raw paths.
- Algorithm should run quickly (under 10 seconds total).
"""

FUNCTION_SIGNATURE = """
import networkx as nx
from typing import List

def search_algorithm(src: str, dsts: List[str], G: nx.DiGraph, num_partitions: int):
    '''
    Find optimal broadcast topology from src to all destinations.

    Args:
        src: Source node (cloud region)
        dsts: List of destination nodes
        G: NetworkX DiGraph with 'cost' and 'throughput' edge attributes
        num_partitions: Number of data partitions

    Returns:
        Broadcast topology object with paths for each (dst, partition) pair
    '''
    pass
"""

SEED_PROGRAM = '''
"""Broadcast optimization algorithm for minimizing transfer cost across multi-cloud networks.

ONLY evolve `search_algorithm` below. `BroadCastTopology` is provided by the
environment (already in scope) — do NOT redefine it, rename it, or write your
own topology class. Just construct it with `BroadCastTopology(src, dsts,
num_partitions)`, fill it via `append_dst_partition_path(...)`, and return it.
Every (dst, partition) must get a continuous path of valid G edges from src to dst.
"""

import networkx as nx
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    # `BroadCastTopology` is supplied by the environment — use it as-is.
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

'''



# --- Evaluation ---

LOWER_COST = 1199.00
UPPER_COST = 626.24
NUM_VMS = 2

CONFIG_NAMES = [
    "intra_aws",
    "intra_azure",
    "intra_gcp",
    "inter_agz",
    "inter_gaz2",
]

_CONTEXT_CACHE: dict[str, Any] | None = None

# Cloudcast is fully self-contained: the simulator modules (simulator.py,
# utils.py, broadcast.py) and the dataset (examples/config/*.json,
# profiles/*.csv) are committed next to this file, so the example runs on CI
# with no external clone and no ADRS_EXAMPLE_DATA_ROOT.
EXAMPLE_DIR = Path(__file__).resolve().parent


def _import_cloudcast_modules() -> tuple[Any, Any, Any]:
    """Import the bundled Cloudcast simulator modules from EXAMPLE_DIR."""
    example_str = str(EXAMPLE_DIR)
    if example_str not in sys.path:
        sys.path.insert(0, example_str)
    try:
        simulator_module = importlib.import_module("simulator")
        utils_module = importlib.import_module("utils")
        broadcast_module = importlib.import_module("broadcast")
    except Exception as e:
        raise ImportError(f"Failed to import Cloudcast modules: {e}") from e

    return (
        simulator_module.BCSimulator,
        utils_module.make_nx_graph,
        broadcast_module.BroadCastTopology,
    )


def _load_context() -> dict[str, Any]:
    """Load and cache Cloudcast simulator resources bundled in EXAMPLE_DIR."""
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is not None:
        return _CONTEXT_CACHE

    BCSimulator, make_nx_graph, BroadCastTopology = _import_cloudcast_modules()

    profiles_dir = EXAMPLE_DIR / "profiles"
    config_dir = EXAMPLE_DIR / "examples" / "config"
    cost_csv = profiles_dir / "cost.csv"
    throughput_csv = profiles_dir / "throughput.csv"

    config_files = {name: config_dir / f"{name}.json" for name in CONFIG_NAMES}

    missing = [str(path) for path in [cost_csv, throughput_csv, *config_files.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Cloudcast files: {missing}")

    graph = make_nx_graph(
        cost_path=str(cost_csv),
        throughput_path=str(throughput_csv),
        num_vms=NUM_VMS,
    )
    configs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in config_files.items()
    }

    _CONTEXT_CACHE = {
        "graph": graph,
        "configs": configs,
        "BCSimulator": BCSimulator,
        "BroadCastTopology": BroadCastTopology,
    }
    return _CONTEXT_CACHE


def _compute_config_score(cost: float) -> float:
    """Convert a per-config cost to a normalized [0, 1] score."""
    per_config_baseline = LOWER_COST / len(CONFIG_NAMES)
    per_config_optimal = UPPER_COST / len(CONFIG_NAMES)
    if cost >= per_config_baseline:
        return 0.0
    if cost <= per_config_optimal:
        return 1.0
    return (per_config_baseline - cost) / (per_config_baseline - per_config_optimal)


def _validate_broadcast_topology(
    bc_t: Any,
    source_node: str,
    terminal_nodes: list[str],
    num_partitions: int,
    G: Any,
) -> tuple[bool, str | None]:
    """
    Validate that a broadcast topology is complete and correct.

    Ported verbatim from the canonical ADRS evaluator
    (benchmarks/ADRS/cloudcast/evaluator/evaluator.py::validate_broadcast_topology)
    so this example scores identically to the leaderboard. Without this gate a
    degenerate topology (missing destinations, empty/partial partitions, or
    discontinuous paths) is silently costed by the simulator on whatever few
    edges it does contain, yielding an artificially tiny cost (e.g. ~189) and a
    bogus top score. The simulator does NOT check coverage, so the check must
    live here.

    Returns (is_valid, error_message).
    """
    # Check 1: all destinations present, no extras.
    if set(bc_t.dsts) != set(terminal_nodes):
        missing_dsts = set(terminal_nodes) - set(bc_t.dsts)
        extra_dsts = set(bc_t.dsts) - set(terminal_nodes)
        return False, f"Destination mismatch: missing={missing_dsts}, extra={extra_dsts}"

    # Check 2: source matches.
    if bc_t.src != source_node:
        return False, f"Source mismatch: expected={source_node}, got={bc_t.src}"

    # Check 3 & 4: every (dst, partition) exists, is non-empty, and forms a
    # continuous route from source to destination over valid graph edges.
    missing_partitions: list[tuple[str, int]] = []
    empty_partitions: list[tuple[str, int]] = []
    invalid_paths: list[tuple[str, int, str]] = []

    for dst in terminal_nodes:
        if dst not in bc_t.paths:
            return False, f"Missing destination '{dst}' in paths"

        for partition_id in range(num_partitions):
            partition_key = str(partition_id)

            if partition_key not in bc_t.paths[dst]:
                missing_partitions.append((dst, partition_id))
                continue

            partition_paths = bc_t.paths[dst][partition_key]
            if partition_paths is None or len(partition_paths) == 0:
                empty_partitions.append((dst, partition_id))
                continue

            path_nodes = [source_node]
            path_valid = True
            for edge in partition_paths:
                if len(edge) < 3:
                    invalid_paths.append((dst, partition_id, "edge format invalid"))
                    path_valid = False
                    break

                edge_src, edge_dst = edge[0], edge[1]
                if not G.has_edge(edge_src, edge_dst):
                    invalid_paths.append((dst, partition_id, f"edge {edge_src}->{edge_dst} not in graph"))
                    path_valid = False
                    break

                if path_nodes[-1] != edge_src:
                    invalid_paths.append((dst, partition_id, f"path discontinuity: expected {path_nodes[-1]}, got {edge_src}"))
                    path_valid = False
                    break

                path_nodes.append(edge_dst)

            if path_valid and path_nodes[-1] != dst:
                invalid_paths.append((dst, partition_id, f"path does not reach destination: ends at {path_nodes[-1]}, expected {dst}"))

    errors: list[str] = []
    if missing_partitions:
        errors.append(f"Missing partitions: {missing_partitions}")
    if empty_partitions:
        errors.append(f"Empty partitions: {empty_partitions}")
    if invalid_paths:
        errors.append(f"Invalid paths: {invalid_paths}")
    if errors:
        return False, "Validation failed: " + "; ".join(errors)

    # Check 5: no data loss — every (dst, partition) pair is actually present.
    expected_total_partitions = len(terminal_nodes) * num_partitions
    actual_partitions = 0
    for dst in terminal_nodes:
        for partition_id in range(num_partitions):
            partition_key = str(partition_id)
            if (
                partition_key in bc_t.paths[dst]
                and bc_t.paths[dst][partition_key] is not None
                and len(bc_t.paths[dst][partition_key]) > 0
            ):
                actual_partitions += 1
    if actual_partitions != expected_total_partitions:
        return False, f"Data loss detected: expected {expected_total_partitions} partitions, got {actual_partitions}"

    return True, None


def _inject_runtime_globals(search_algorithm: Any, broad_cast_topology_cls: Any) -> list[str]:
    """Inject common globals so candidate code can run without boilerplate imports."""
    runtime_globals = search_algorithm.__globals__
    injections = {
        "BroadCastTopology": broad_cast_topology_cls,
        "nx": nx,
        "networkx": nx,
        "np": np,
        "numpy": np,
        "math": math,
        "time": time_module,
        "random": random,
        "collections": collections,
        "heapq": heapq,
        "Dict": Dict,
        "List": List,
        "Set": Set,
        "Tuple": Tuple,
        "Any": Any,
    }

    added_keys: list[str] = []
    for key, value in injections.items():
        if key not in runtime_globals:
            runtime_globals[key] = value
            added_keys.append(key)
    return added_keys


def _restore_runtime_globals(search_algorithm: Any, added_keys: list[str]) -> None:
    runtime_globals = search_algorithm.__globals__
    for key in added_keys:
        runtime_globals.pop(key, None)


def score_fn(search_algorithm: Any, _inputs: list[Any] | None = None) -> dict:
    """
    Evaluate a Cloudcast search algorithm and return a 0-100 score.

    Returns score plus per-config metrics used as behavior dimensions.
    """
    try:
        context = _load_context()
    except Exception as e:
        return {"error": f"Cloudcast setup error: {e}"}

    added_keys = _inject_runtime_globals(search_algorithm, context["BroadCastTopology"])

    per_config_costs: dict[str, float] = {}
    per_config_times: dict[str, float] = {}
    per_config_scores: dict[str, float] = {}
    total_cost = 0.0
    total_time = 0.0

    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            for config_name, config in context["configs"].items():
                try:
                    bc_topology = search_algorithm(
                        config["source_node"],
                        config["dest_nodes"],
                        context["graph"],
                        config["num_partitions"],
                    )
                    bc_topology.set_num_partitions(config["num_partitions"])

                    # Reject incomplete / discontinuous topologies BEFORE costing,
                    # matching the canonical ADRS evaluator. The simulator only
                    # costs the edges it is handed, so without this gate a
                    # partial topology gets a bogus tiny cost (the ~189 score-100
                    # artifact). A failed config is a hard reject for the whole
                    # candidate, exactly as the leaderboard evaluator does.
                    is_valid, validation_error = _validate_broadcast_topology(
                        bc_topology,
                        config["source_node"],
                        config["dest_nodes"],
                        config["num_partitions"],
                        context["graph"],
                    )
                    if not is_valid:
                        return {"error": f"Invalid broadcast topology for {config_name}: {validation_error}"}

                    simulator = context["BCSimulator"](num_vms=NUM_VMS, output_dir="evals")
                    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
                        transfer_time, cost = simulator.evaluate_path(bc_topology, config)
                except Exception as e:
                    return {"error": f"Config {config_name} failed: {str(e)[:160]}"}

                if not math.isfinite(cost) or cost < 0:
                    return {"error": f"Invalid cost for {config_name}: {cost}"}
                if not math.isfinite(transfer_time) or transfer_time <= 0:
                    return {"error": f"Invalid transfer time for {config_name}: {transfer_time}"}

                per_config_costs[config_name] = float(cost)
                per_config_times[config_name] = float(transfer_time)
                per_config_scores[config_name] = _compute_config_score(float(cost))
                total_cost += float(cost)
                total_time += float(transfer_time)
    finally:
        os.chdir(original_cwd)
        _restore_runtime_globals(search_algorithm, added_keys)

    if total_cost <= 0:
        return {"error": "Invalid solution: zero total cost (likely no data transferred)"}
    if not math.isfinite(total_time) or total_time <= 0:
        return {"error": f"Invalid total transfer time: {total_time}"}

    cost_clamped = max(min(total_cost, LOWER_COST), UPPER_COST)
    normalized_cost = (LOWER_COST - cost_clamped) / (LOWER_COST - UPPER_COST)
    score = normalized_cost * 100.0

    return {
        "score": float(score),
        "total_cost": float(total_cost),
        "total_time": float(total_time),
        "successful_configs": len(per_config_costs),
        "per_config_costs": per_config_costs,
        "per_config_times": per_config_times,
        "intra_aws_score": per_config_scores.get("intra_aws", 0.0),
        "intra_azure_score": per_config_scores.get("intra_azure", 0.0),
        "intra_gcp_score": per_config_scores.get("intra_gcp", 0.0),
        "inter_agz_score": per_config_scores.get("inter_agz", 0.0),
        "inter_gaz2_score": per_config_scores.get("inter_gaz2", 0.0),
    }