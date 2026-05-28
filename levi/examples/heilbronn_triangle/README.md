# Heilbronn Triangle Example (n = 11)

Place 11 points inside the unit equilateral triangle so that the smallest
triangle formed by any three of them has maximum area. Mirrors the setup in
`benchmarks/math/heilbronn_triangle`.

Candidate function API:

```python
def heilbronn_triangle11() -> np.ndarray:
    # returns points of shape (11, 2)
```

`score_fn` rejects:
- arrays not of shape `(11, 2)` or containing non-finite values,
- points outside the equilateral triangle.

Valid solutions get `score = min_triangle_area / unit_triangle_area`, and
`combined_score = score / 0.0365298898800301...` (AlphaEvolve benchmark).

## Run

```bash
cd levi/examples/heilbronn_triangle
uv run python run.py --evals 20
```

Models default to BLADE's standard split (Qwen-30B mutation, GPT-5 paradigm,
OpenAI embedding). Override via `--mutation-model`, `--paradigm-model`,
`--embedding-model`, or environment variables `BLADE_MUTATION_MODEL`,
`BLADE_PARADIGM_MODEL`, `BLADE_EMBEDDING_MODEL`.
