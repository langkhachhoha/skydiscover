"""Compute SWDI and CDI (HSEvo, Pham et al. 2024) for checkpointed populations.

Pipeline follows Section 3 of the HSEvo paper:
  (i)   strip comments/docstrings with the abstract-syntax tree,
  (ii)  standardise formatting into a single canonical coding style,
  (iii) embed each snippet with a code/text embedding model.

SWDI: greedy clustering under cosine similarity threshold alpha, then Shannon
entropy over the cluster-size distribution.
CDI:  minimum spanning tree over Euclidean distances, then Shannon entropy over
the normalised MST edge-length distribution.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "openai/text-embedding-3-small"
ALPHA = 0.95  # cosine similarity threshold for SWDI clustering


# --------------------------------------------------------------------------- #
# (i) + (ii) population encoding: strip docstrings/comments, canonicalise style
# --------------------------------------------------------------------------- #
class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:] or [ast.Pass()]
            node.body = body
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


def normalize_code(code: str) -> str:
    """Remove comments/docstrings and re-emit in one canonical style."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Unparseable individual: fall back to whitespace-normalised source.
        return "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)  # comments are already gone; output is deterministic


# --------------------------------------------------------------------------- #
# (iii) embeddings
# --------------------------------------------------------------------------- #
def embed(texts, api_key, cache_path, batch_size=32):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    # de-duplicate within the request set
    todo = list({k: t for k, t in todo}.items())

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        for attempt in range(5):
            resp = requests.post(
                EMBED_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": EMBED_MODEL, "input": [t for _, t in batch]},
                timeout=120,
            )
            if resp.status_code == 200:
                break
            print(f"  retry {attempt + 1}: HTTP {resp.status_code} {resp.text[:200]}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("embedding request failed after 5 attempts")

        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        for (k, _), item in zip(batch, data):
            cache[k] = item["embedding"]
        print(f"  embedded {min(i + batch_size, len(todo))}/{len(todo)} new snippets")
        cache_path.write_text(json.dumps(cache))

    return np.array([cache[k] for k in keys], dtype=np.float64)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def shannon(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def swdi(vectors, alpha=ALPHA):
    """Greedy clustering: v_i joins C_i only if it is >= alpha similar to ALL members."""
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    clusters = []  # list of lists of row indices
    for i, v in enumerate(normed):
        placed = False
        for c in clusters:
            if all(float(v @ normed[k]) >= alpha for k in c):
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    sizes = np.array([len(c) for c in clusters], dtype=np.float64)
    return shannon(sizes / sizes.sum()), len(clusters)


def cdi(vectors):
    """Shannon entropy of the normalised edge lengths of the Euclidean MST."""
    dist = squareform(pdist(vectors, metric="euclidean"))
    mst = minimum_spanning_tree(dist)
    edges = mst.toarray()
    d = edges[edges > 0]
    if d.size == 0:
        return 0.0, 0
    return shannon(d / d.sum()), int(d.size)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="diversity/circle_packing_rect")
    ap.add_argument("--variants", nargs="+", default=["multi_prompt", "single_prompt"])
    ap.add_argument("--out", default="diversity/circle_packing_rect/diversity_results.txt")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--dedup", action="store_true",
                    help="drop duplicate code snippets within a checkpoint before scoring")
    args = ap.parse_args()

    api_key = None
    for line in Path(".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            break
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("no API key found in .env")

    root = Path(args.root)
    cache_path = root / ".embedding_cache.json"
    results = {}

    for variant in args.variants:
        results[variant] = []
        for ckpt in sorted((root / variant / "checkpoints").glob("checkpoint_*.json")):
            payload = json.loads(ckpt.read_text())
            codes = [ind["code"] for ind in payload["population"]]
            snippets = [normalize_code(c) for c in codes]
            if args.dedup:
                snippets = list(dict.fromkeys(snippets))
            print(f"{variant}/{ckpt.name}: {len(snippets)} individuals")

            vecs = embed(snippets, api_key, cache_path)
            s, n_clusters = swdi(vecs, args.alpha)
            c, n_edges = cdi(vecs)
            results[variant].append(
                {
                    "checkpoint": payload["checkpoint"],
                    "n": len(snippets),
                    "n_clusters": n_clusters,
                    "n_mst_edges": n_edges,
                    "swdi": s,
                    "cdi": c,
                    "best_score": payload["window"].get("best_score"),
                    "eval_end": payload["window"].get("eval_end"),
                }
            )

    out = Path(args.out)
    lines = [
        f"SWDI / CDI diversity analysis - {root.name}",
        "Metrics follow HSEvo (Pham Vu Tuan Dat et al., 2024), Section 3.",
        f"Embedding model : {EMBED_MODEL} (via OpenRouter)",
        f"SWDI threshold  : alpha = {args.alpha}",
        "Encoding        : AST comment/docstring removal + canonical re-emission (ast.unparse)",
        "Population      : the archive snapshot stored in each checkpoint"
        + (" (duplicates removed)" if args.dedup else " (as-is, duplicates kept)"),
        "SWDI higher = more even spread across clusters; CDI higher = more dispersed population.",
        "",
    ]
    for variant, rows in results.items():
        lines.append(f"=== {variant} ===")
        lines.append(
            f"{'ckpt':>5} {'N':>4} {'#clusters':>10} {'SWDI':>8} {'CDI':>8} {'best_score':>12} {'evals':>7}"
        )
        for r in rows:
            bs = f"{r['best_score']:.6f}" if r["best_score"] is not None else "-"
            lines.append(
                f"{r['checkpoint']:>5} {r['n']:>4} {r['n_clusters']:>10} "
                f"{r['swdi']:>8.4f} {r['cdi']:>8.4f} {bs:>12} {r['eval_end']:>7}"
            )
        sw = np.array([r["swdi"] for r in rows])
        cd = np.array([r["cdi"] for r in rows])
        lines.append(f"{'mean':>5} {'':>4} {'':>10} {sw.mean():>8.4f} {cd.mean():>8.4f}")
        lines.append(f"{'std':>5} {'':>4} {'':>10} {sw.std(ddof=1):>8.4f} {cd.std(ddof=1):>8.4f}")
        lines.append("")

    out.write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
