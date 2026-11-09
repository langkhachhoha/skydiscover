"""coral_tools.py: reading CORAL attempts and pricing the gateway's request log."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import coral_tools as T  # noqa: E402

GPT5 = "openrouter/openai/gpt-5"


def _entry(status=200, usage=None, path="/v1/chat/completions", ms=2000):
    resp = {"choices": []}
    if usage is not None:
        resp["usage"] = usage
    return {"path": path, "status_code": status, "duration_ms": ms, "response": resp}


@pytest.fixture
def job(tmp_path: Path) -> Path:
    job = tmp_path / "circle_packing"
    coral = job / "results" / "circle-packing" / "20261108-120000" / ".coral" / "public"
    (coral / "attempts").mkdir(parents=True)
    (coral / "gateway").mkdir(parents=True)
    for h, score, status in [("a1", 0.95, "improved"), ("a2", 0.99, "improved"), ("a3", None, "crashed"),
                             ("a4", None, "pending")]:
        (coral / "attempts" / f"{h}.json").write_text(json.dumps(
            {"commit_hash": h, "score": score, "status": status, "title": f"try {h}", "timestamp": "t"}))
    (job / ".started_at").write_text(str(time.time() - 3600))
    (job / ".ended_at").write_text(str(time.time()))
    (job / ".exit_code").write_text("124\n")
    return job


def _write_log(job: Path, entries: list[dict], *, partial: str = "") -> Path:
    log = next(job.glob("results/*/*/.coral/public/gateway")) / "requests.jsonl"
    with log.open("a") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
        fh.write(partial)
    return log


def test_status_prices_tokens_and_reads_attempts(job: Path):
    _write_log(job, [
        _entry(usage={"prompt_tokens": 100_000, "completion_tokens": 2_000,
                      "prompt_tokens_details": {"cached_tokens": 80_000},
                      "completion_tokens_details": {"reasoning_tokens": 1_500}}),
        _entry(usage={"prompt_tokens": 10_000, "completion_tokens": 1_000, "cost": 0.5}),  # provider cost wins
        _entry(usage=None),                       # logged without usage
        _entry(status=429),                       # failed call
        {"path": "/health", "status_code": 200},  # not an LLM call
    ])
    r = T.collect("c1", "circle_packing", job, GPT5, None)
    assert r["state"] == "DONE"
    assert r["attempts"] == 3 and r["crashed"] == 1
    assert r["best_score"] == pytest.approx(0.99)
    assert r["best"] == pytest.approx(0.99 * T.BEST_KNOWN["circle_packing"])
    assert (r["llm_calls"], r["llm_errors"], r["calls_without_usage"]) == (3, 1, 1)
    assert r["input_tokens"] == 110_000 and r["cached_tokens"] == 80_000
    assert r["output_tokens"] == 3_000 and r["reasoning_tokens"] == 1_500
    first = 20_000 * 1.25e-6 + 80_000 * 1.25e-7 + 2_000 * 1e-5
    assert r["cost"] == pytest.approx(first + 0.5)
    assert r["s_per_call"] == pytest.approx(2.0)


def test_usage_is_read_incrementally_and_waits_for_whole_lines(job: Path):
    u = {"prompt_tokens": 1_000, "completion_tokens": 100}
    log = _write_log(job, [_entry(usage=u)], partial='{"path": "/v1/chat/compl')
    coral = T._coral_dir(job)
    t1 = T.gateway_usage(coral, job, GPT5, None)
    assert t1["calls"] == 1
    # The half-written line is completed, and a new call is appended.
    with log.open("a") as fh:
        fh.write('etions", "status_code": 200, "response": {"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}}\n')
        fh.write(json.dumps(_entry(usage=u)) + "\n")
    t2 = T.gateway_usage(coral, job, GPT5, None)
    assert t2["calls"] == 3 and t2["input"] == 3_000
    assert T.gateway_usage(coral, job, GPT5, None)["calls"] == 3   # cached, not double counted


def test_responses_api_usage_shape(job: Path):
    e = _entry(path="/v1/responses", usage=None)
    e["response"] = {"response": {"usage": {"input_tokens": 500, "output_tokens": 50,
                                            "input_tokens_details": {"cached_tokens": 100},
                                            "output_tokens_details": {"reasoning_tokens": 20}}}}
    _write_log(job, [e])
    r = T.collect("c1", "circle_packing", job, GPT5, None)
    assert (r["input_tokens"], r["cached_tokens"], r["output_tokens"], r["reasoning_tokens"]) == (500, 100, 50, 20)


def test_pricing_file_overrides_defaults(job: Path, tmp_path: Path):
    (tmp_path / "pricing.json").write_text(json.dumps({GPT5: {"prompt": 1e-6, "completion": 2e-6}}))
    _write_log(job, [_entry(usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})])
    r = T.collect("c1", "circle_packing", job, GPT5, tmp_path)
    assert r["cost"] == pytest.approx(3.0)


def test_streams_cut_before_usage_are_priced_from_openrouter(job: Path, monkeypatch):
    cut = {"path": "/v1/chat/completions", "status_code": 200, "duration_ms": 900,
           "response": {"id": "gen-123", "model": "gpt-5"}}
    _write_log(job, [cut])
    monkeypatch.setattr(T, "fetch_generation", lambda gid: None)          # not available yet
    r = T.collect("c1", "circle_packing", job, GPT5, None)
    assert r["llm_calls"] == 1 and r["calls_without_usage"] == 1 and r["cost"] == 0
    monkeypatch.setattr(T, "fetch_generation", lambda gid: {
        "input": 40_000.0, "output": 2_000.0, "cached": 39_000.0, "reasoning": 1_000.0, "cost": 0.027})
    r = T.collect("c1", "circle_packing", job, GPT5, None)
    assert r["calls_without_usage"] == 0
    assert r["cost"] == pytest.approx(0.027) and r["reasoning_tokens"] == 1_000
    assert T.collect("c1", "circle_packing", job, GPT5, None)["cost"] == pytest.approx(0.027)  # counted once


def test_pending_and_states(tmp_path: Path):
    assert T.collect("c2", "circle_packing_rect", tmp_path / "nope", GPT5, None)["state"] == "PENDING"
    j = tmp_path / "j"
    j.mkdir()
    (j / ".started_at").write_text(str(time.time()))
    assert T.collect("c2", "circle_packing_rect", j, GPT5, None)["state"] == "STOPPED"
    (j / ".exit_code").write_text("1\n")
    assert T.collect("c2", "circle_packing_rect", j, GPT5, None)["state"] == "FAILED(1)"
    # `stop` makes CORAL exit 0 like a finished run; the marker keeps it from reading as DONE.
    (j / ".exit_code").write_text("0\n")
    (j / ".stopped_by_user").write_text("1")
    assert T.collect("c2", "circle_packing_rect", j, GPT5, None)["state"] == "STOPPED"


def test_count_and_eval_cap(job: Path, tmp_path: Path, capsys):
    assert T.count_graded(job) == 3                      # a4 is still pending
    assert T.count_graded(tmp_path / "nope") == 0
    # The cap's SIGTERM can leave CORAL dead of the signal (143): still a finished run.
    (job / ".exit_code").write_text("143\n")
    assert T.collect("c1", "circle_packing", job, GPT5, None)["state"] == "FAILED(143)"
    (job / ".eval_cap_reached").write_text("1")
    r = T.collect("c1", "circle_packing", job, GPT5, None)
    assert (r["state"], r["ended_by"]) == ("DONE", "evals")
    assert T._evals(r, 100) == "3/100" and T._evals(r, 0) == "3"
    (job / ".eval_cap_reached").unlink()
    (job / ".exit_code").write_text("124\n")
    assert T.collect("c1", "circle_packing", job, GPT5, None)["ended_by"] == "time"


def _attempts(job: Path, rows) -> None:
    d = job / "results" / "x" / "20261109-010000" / ".coral" / "public" / "attempts"
    d.mkdir(parents=True)
    for h, score, status, meta in rows:
        a = {"commit_hash": h, "score": score, "status": status, "title": h, "timestamp": "t"}
        if meta:
            a["metadata"] = meta
        (d / f"{h}.json").write_text(json.dumps(a))
    (job / ".started_at").write_text(str(time.time() - 60))


def test_kernel_builder_best_is_fewest_cycles_and_tune_sweeps_are_left_out(tmp_path: Path):
    job = tmp_path / "kernel_builder"
    _attempts(job, [("k1", 11910.0, "baseline", None), ("k2", 4000.0, "improved", None),
                    ("k3", None, "crashed", None),
                    ("k4", 900.0, "improved", {"budget_class": "tune"})])   # smaller workload: not comparable
    r = T.collect("p2", "kernel_builder", job, GPT5, None)
    assert r["best_score"] == 4000.0                                   # cycles, lower is better
    assert r["best"] == pytest.approx(147734 / 4000)                    # as run_pk_queue.sh reports it
    assert (r["attempts"], r["tune_attempts"], r["crashed"]) == (4, 1, 1)


def test_polyominoes_best_is_the_judge_score(tmp_path: Path):
    job = tmp_path / "polyominoes"
    _attempts(job, [("q1", 0.0, "baseline", None), ("q2", 61.2, "improved", None), ("q3", 55.0, "regressed", None)])
    r = T.collect("p1", "polyominoes", job, GPT5, None)
    assert r["best_score"] == r["best"] == pytest.approx(61.2)


def _fake_coral_task(root: Path, rel: str, task_yaml: str, files: dict[str, str]) -> None:
    src = root / rel
    for name, text in {"task.yaml": task_yaml, **files}.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(text)


def _prepare(tmp_path: Path, task: str, coral: Path, frontier: Path) -> tuple[Path, dict]:
    import argparse

    import yaml

    job = tmp_path / "job" / task
    T.cmd_prepare(argparse.Namespace(task=task, job_dir=str(job), coral_dir=str(coral), port=4103, model=GPT5,
                                     agents=1, context=400000, max_output=128000,
                                     frontier_cs_dir=str(frontier), judge_workers=3))
    return job, yaml.safe_load((job / "task.yaml").read_text())


def test_prepare_polyominoes_swaps_in_the_local_judge_and_frontier_statement(tmp_path: Path):
    coral, frontier = tmp_path / "CORAL", tmp_path / "Frontier-CS"
    _fake_coral_task(coral, "examples/frontier_cs_algo/0", """
task: {name: poly, description: d, tips: t}
grader:
  entrypoint: "frontier_cs_algo_grader.grader:Grader"
  setup: ["uv pip install -e ./grader", "uv pip install git+https://github.com/FrontierCS/Frontier-CS.git"]
  timeout: 340
  direction: maximize
  args: {problem_id: '0', n_cases: 70}
agents: {count: 1, runtime: claude_code, model: claude-sonnet-4-6, max_turns: 200}
workspace: {results_dir: ./results, repo_path: ./examples/frontier_cs_algo/0/seed}
run: {session: tmux}
""", {"grader/src/frontier_cs_algo_grader/grader.py": "from frontier_cs import SingleEvaluator\n",
      "seed/statement.txt": "old statement, W=H\n", "seed/solution.cpp": "int main(){}\n"})
    (frontier / "algorithmic/problems/0").mkdir(parents=True)
    (frontier / "algorithmic/problems/0/chk.cc").write_text("// checker\n")
    (frontier / "algorithmic/problems/0/statement.txt").write_text("current statement\n")
    job, cfg = _prepare(tmp_path, "polyominoes", coral, frontier)
    pkg = job / "grader/src/frontier_cs_algo_grader"
    assert "SingleEvaluator" not in (pkg / "grader.py").read_text() and "from . import judge" in (pkg / "grader.py").read_text()
    assert (pkg / "judge.py").read_text() == T.POLYOMINO_JUDGE.read_text()
    assert (job / "seed/statement.txt").read_text() == "current statement\n"
    assert cfg["grader"]["setup"] == ["uv pip install -e ./grader"]
    assert cfg["grader"]["args"]["frontier_cs_dir"] == str(frontier.resolve())
    assert cfg["agents"]["runtime"] == "opencode" and cfg["agents"]["gateway"]["port"] == 4103
    assert cfg["run"]["session"] == "local" and cfg["workspace"]["repo_path"] == str(job / "seed")


def test_prepare_kernel_builder_keeps_private_data_and_the_presets_grader_install(tmp_path: Path):
    coral = tmp_path / "CORAL"
    _fake_coral_task(coral, "examples/kernel_builder", """
task: {name: kb, description: d, tips: t}
preset: multi-island
grader:
  entrypoint: "kernel_builder_grader.grader:Grader"
  timeout: 120
  direction: minimize
  private: ["taskdata"]
  args: {program_file: "kernel_builder.py"}
workspace: {results_dir: ./results, repo_path: ./examples/kernel_builder/seed}
run: {session: local}
""", {"grader/pyproject.toml": "", "seed/kernel_builder.py": "class KernelBuilder: pass\n",
      "taskdata/frozen_problem.py": "# simulator\n"})
    job, cfg = _prepare(tmp_path, "kernel_builder", coral, tmp_path / "unused")
    assert (job / "taskdata/frozen_problem.py").exists()      # private paths resolve next to task.yaml
    assert "preset" not in cfg and "islands" not in cfg
    assert cfg["grader"]["setup"] == ["uv pip install -e ./grader"]
    assert cfg["grader"]["direction"] == "minimize" and cfg["agents"]["count"] == 1
