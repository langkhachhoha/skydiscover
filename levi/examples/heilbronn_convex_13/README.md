# Heilbronn Convex Example (n = 13)

Place 13 points in the plane so that the smallest triangle area (normalized
by the convex-hull area of all 13 points) is maximized. Mirrors the setup in
`benchmarks/math/heilbronn_convex/13`.

Candidate function API:

```python
def heilbronn_convex13() -> np.ndarray:
    # returns points of shape (13, 2)
```

`score_fn` rejects:
- arrays not of shape `(13, 2)` or containing non-finite values,
- point sets whose convex hull is degenerate (zero area).

Valid solutions get `score = min_triangle_area / convex_hull_area`, and
`combined_score = score / 0.0309368890348956...` (AlphaEvolve benchmark).
The score is scale-invariant, so coordinate ranges do not matter.

## Run

```bash
cd levi/examples/heilbronn_convex_13
uv run python run.py --evals 20
```

Models default to BLADE's standard split (Qwen-30B mutation, GPT-5 paradigm,
OpenAI embedding). Override via `--mutation-model`, `--paradigm-model`,
`--embedding-model`, or environment variables `BLADE_MUTATION_MODEL`,
`BLADE_PARADIGM_MODEL`, `BLADE_EMBEDDING_MODEL`.
