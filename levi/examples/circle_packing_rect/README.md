# Circle Packing Rect Example

This mirrors the benchmark setup in `benchmarks/math/circle_packing_rect`:

- `n = 21` circles
- rectangle perimeter `4`, so `width + height <= 2`
- objective: maximize `sum(circles[:, 2])`
- candidate API: `circle_packing21() -> np.ndarray` with shape `(21, 3)`

The rectangle is inferred from the returned circles' minimum circumscribing
rectangle. Invalid packings are rejected.

BLADE bootstraps the initial programs from the frontier model; no starter
implementation is bundled with this example.

## Run

From the repository root:

```bash
uv --project levi run python scripts/run_blade.py \
    --example-dir levi/examples/circle_packing_rect \
    --evals 20
```

Or run the example-local driver:

```bash
uv --project levi run python levi/examples/circle_packing_rect/run.py --evals 20
```

The default BLADE models are:

- mutation: `openrouter/qwen/qwen3-30b-a3b-instruct-2507`
- paradigm: `openrouter/openai/gpt-5`
- embedding: `openrouter/openai/text-embedding-3-small`

API keys are loaded from the repository `.env` when present.
