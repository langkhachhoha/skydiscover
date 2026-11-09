#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_pk_queue.sh — CORAL's two stress-test tasks, Polyominoes and Kernel
# Engineering, run by the four baselines and by SpecEvo: two tmux lanes side
# by side, one per task, each running the five methods one after the other.
#
#   ./scripts/server/run_pk_queue.sh check                 # offline pre-flight: data, judge, seeds, commands (no API calls)
#   ./scripts/server/run_pk_queue.sh start   [poly kern]   # launch lanes (default: both), detached
#   ./scripts/server/run_pk_queue.sh status  [poly kern]   # done / running (live numbers) / left
#   ./scripts/server/run_pk_queue.sh compare [poly kern]   # tokens / reasoning / latency / score table (default: both)
#   ./scripts/server/run_pk_queue.sh stop    [poly kern]   # kill lanes and their runs
#   DRY_RUN=1 ./scripts/server/run_pk_queue.sh start       # print the 10 commands, no API calls
#
# Lanes (tmux session — task), each running these methods in this order:
#   poly  pk_poly    polyominoes
#   kern  pk_kernel  kernel_builder
#     openevolve_native, gepa_native, adaevolve, evox   skydiscover baselines on $BASELINE_MODEL
#                                                       (gpt-5), up to $ITERATIONS iterations -- the
#                                                       time / spend budget is what stops them
#     specevo                                           qwen3-30b Speculator + gpt-5 Navigator,
#                                                       $WORKERS workers
#
# Every job stops at whichever comes first: $SECONDS_CAP s (3h) or $DOLLARS
# ($20) of tracked LLM spend, both graceful (best program and summary
# written); $HARD_TIMEOUT kills a run that does not stop by itself. A lane is
# 5 jobs, so up to ~15h.
#
# Tasks (benchmarks/coral_tasks, levi/examples/{polyominoes,kernel_builder}):
#   polyominoes     Frontier-CS algorithmic #0, pack all polyominoes into the smallest rectangle;
#                   C++17, judged locally (2 s CPU / 256 MB per case, 70 cases), score 0..100
#                   (CORAL paper: 4 agents 84.2 w/o web, SOTA 87.0). Needs a Frontier-CS
#                   checkout: scripts/server/setup_coral.sh.
#   kernel_builder  Anthropic's VLIW SIMD take-home, score = 147734 / cycles (official best
#                   1363 cycles = 108.4; CORAL 4 agents 1103 = 133.9)
#
# Outputs: outputs/server/$GROUP/<task>/<task>__<method>/
# Restarting is safe: a finished job is skipped; one that died half-way is
# moved aside to <dir>.incomplete_<ts> and run again.
#
# Knobs (environment variables, defaults in brackets):
#   GROUP [pk_exp]  SECONDS_CAP [10800]  DOLLARS [20]  HARD_TIMEOUT [11700]  ITERATIONS [1000]
#   BASELINE_MODEL / PARADIGM_MODEL [openrouter/openai/gpt-5]
#   MUTATION_MODEL [openrouter/qwen/qwen3-30b-a3b-instruct-2507]
#   WORKERS [4]  PE_INTERVAL [10]  EVAL_TIMEOUT [600]  THREADS_PER_EVAL [2]
#   JUDGE_WORKERS [3]  (Polyominoes test cases judged in parallel per evaluation)
#   FRONTIER_CS_DIR [/home/ubuntu/projects/Frontier-CS]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
SELF="$REPO_ROOT/scripts/server/run_pk_queue.sh"

GROUP="${GROUP:-pk_exp}"
SECONDS_CAP="${SECONDS_CAP:-10800}"
DOLLARS="${DOLLARS:-20}"
HARD_TIMEOUT="${HARD_TIMEOUT:-11700}"
ITERATIONS="${ITERATIONS:-1000}"
BASELINE_MODEL="${BASELINE_MODEL:-openrouter/openai/gpt-5}"
PARADIGM_MODEL="${PARADIGM_MODEL:-openrouter/openai/gpt-5}"
MUTATION_MODEL="${MUTATION_MODEL:-openrouter/qwen/qwen3-30b-a3b-instruct-2507}"
WORKERS="${WORKERS:-4}"
PE_INTERVAL="${PE_INTERVAL:-10}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"
THREADS_PER_EVAL="${THREADS_PER_EVAL:-2}"
JUDGE_WORKERS="${JUDGE_WORKERS:-3}"
FRONTIER_CS_DIR="${FRONTIER_CS_DIR:-/home/ubuntu/projects/Frontier-CS}"

KNOBS=(GROUP SECONDS_CAP DOLLARS HARD_TIMEOUT ITERATIONS BASELINE_MODEL PARADIGM_MODEL MUTATION_MODEL
       WORKERS PE_INTERVAL EVAL_TIMEOUT THREADS_PER_EVAL JUDGE_WORKERS FRONTIER_CS_DIR)

DRY_RUN="${DRY_RUN:-0}"
GROUP_DIR="outputs/server/$GROUP"
[[ "$DRY_RUN" == 1 ]] && GROUP_DIR="outputs/server/${GROUP}_dryrun"
VENV="$REPO_ROOT/.venv"

LANES=(poly kern)
METHODS=(openevolve_native gepa_native adaevolve evox specevo)   # run order inside a lane

die() { echo "ERROR: $*" >&2; exit 1; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }

lane_session() { case "$1" in poly) echo pk_poly ;; kern) echo pk_kernel ;; *) die "unknown lane '$1' (poly | kern)" ;; esac; }
lane_task()    { case "$1" in poly) echo polyominoes ;; kern) echo kernel_builder ;; esac; }
lane_dir()     { echo "$GROUP_DIR/$(lane_task "$1")"; }

lane_jobs() {   # lane -> "<lane> <task> <method>" lines, in run order
    local method
    for method in "${METHODS[@]}"; do echo "$1 $(lane_task "$1") $method"; done
}

job_kind() { [[ "$1" == specevo ]] && echo blade || echo baseline; }    # method -> kind
job_dir()  { echo "$(lane_dir "$1")/$2__$3"; }

job_done() {    # lane task method -> 0 when the run left its final result
    local d; d="$(job_dir "$1" "$2" "$3")"
    if [[ "$(job_kind "$3")" == blade ]]; then [[ -f "$d/summary.json" ]]
    else [[ -f "$d/run/best/best_program_info.json" ]]; fi
}

build_cmd() {   # lane task method -> global array CMD
    local lane="$1" task="$2" method="$3" out run_id bench="$REPO_ROOT/scripts/server/run_bench.sh"
    out="$(job_dir "$lane" "$task" "$method")"
    run_id="${GROUP}_${task}_${method}"
    if [[ "$(job_kind "$method")" == baseline ]]; then
        CMD=("$bench" baseline --no-conda --no-install-deps
             --baseline "$method" --benchmark-dir "benchmarks/coral_tasks/$task"
             --model "$BASELINE_MODEL" --iterations "$ITERATIONS"
             --seconds "$SECONDS_CAP" --timeout "$HARD_TIMEOUT" --dollars "$DOLLARS"
             --run-id "$run_id" --output-dir "$out")
    else
        CMD=("$bench" blade --no-conda --no-install-deps
             --benchmark "levi/examples/$task" --ablation full
             --seconds "$SECONDS_CAP" --timeout "$HARD_TIMEOUT" --dollars "$DOLLARS"
             --mutation-model "$MUTATION_MODEL" --paradigm-model "$PARADIGM_MODEL"
             --workers "$WORKERS" --pe-interval "$PE_INTERVAL" --eval-timeout "$EVAL_TIMEOUT"
             --save-eval-code --run-id "$run_id" --output-dir "$out")
    fi
    [[ "$DRY_RUN" == 1 ]] && CMD+=(--dry-run)
    return 0
}

run_job() {     # lane task method
    local lane="$1" task="$2" method="$3" out rc
    out="$(job_dir "$lane" "$task" "$method")"
    if job_done "$lane" "$task" "$method"; then
        echo "[$(ts)] skip   $task / $method (already finished)"
        return 0
    fi
    if [[ -d "$out" && "$DRY_RUN" != 1 ]]; then
        local aside="$out.incomplete_$(date +%Y%m%d-%H%M%S)"
        mv "$out" "$aside"
        echo "[$(ts)] moved  half-finished $out -> $aside"
    fi
    build_cmd "$lane" "$task" "$method"
    echo "[$(ts)] start  $task / $method -> $out"
    set +e
    "${CMD[@]}" < /dev/null
    rc=$?
    set -e
    [[ "$DRY_RUN" == 1 ]] || echo "$rc" > "$out/.exit_code"
    echo "[$(ts)] end    $task / $method  exit=$rc  $(job_done "$lane" "$task" "$method" && echo finished || echo 'NO RESULT')"
    return 0
}

lane_env() {
    export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
    export OMP_NUM_THREADS="$THREADS_PER_EVAL" OPENBLAS_NUM_THREADS="$THREADS_PER_EVAL" \
           MKL_NUM_THREADS="$THREADS_PER_EVAL" NUMEXPR_NUM_THREADS="$THREADS_PER_EVAL"
    export POLYOMINO_JUDGE_WORKERS="$JUDGE_WORKERS" FRONTIER_CS_DIR
}

lane_main() {
    local lane="$1" l task method
    lane_env
    mkdir -p "$(lane_dir "$lane")"
    echo "[$(ts)] lane $lane begins: $(lane_task "$lane"), ${#METHODS[@]} methods one after another, ${SECONDS_CAP}s or \$${DOLLARS} per job"
    while read -r l task method; do
        run_job "$l" "$task" "$method"
    done < <(lane_jobs "$lane")
    echo "[$(ts)] lane $lane finished."
}

# ---------------------------------------------------------------------------
pick_lanes() {
    local l
    if [[ $# -eq 0 ]]; then echo "${LANES[*]}"; return; fi
    for l in "$@"; do lane_session "$l" >/dev/null; done
    echo "$*"
}

report_jobs() {
    local lane l task method
    for lane in "$@"; do
        while read -r l task method; do
            echo "$l $task $method $(job_kind "$method") $REPO_ROOT/$(job_dir "$l" "$task" "$method")"
        done < <(lane_jobs "$lane")
    done
}

py() { if [[ -x "$VENV/bin/python" ]]; then "$VENV/bin/python" "$@"; else python3 "$@"; fi; }

cmd_status() {
    local lanes s l
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    report_jobs "${lanes[@]}" | py "$REPO_ROOT/scripts/server/cp_report.py" status
    echo
    echo "polyominoes: best = score 0..100.   kernel_builder: best = 147734 / cycles (cycles = 147734 / best)."
    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux has-session -t "=$s" 2>/dev/null && echo "tmux $s: alive" || echo "tmux $s: not running"
    done
}

cmd_compare() {
    local lanes
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    mkdir -p "$GROUP_DIR"
    report_jobs "${lanes[@]}" | py "$REPO_ROOT/scripts/server/cp_report.py" compare --out "$REPO_ROOT/$GROUP_DIR"
}

cmd_stop() {
    local lanes l s task method d p
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux kill-session -t "=$s" 2>/dev/null && echo "killed tmux $s" || true
        while read -r _ task method; do
            d="$(job_dir "$l" "$task" "$method")"
            if [[ -s "$d/.run.pid" ]] && kill -0 "$(cat "$d/.run.pid")" 2>/dev/null; then
                p="$(cat "$d/.run.pid")"
                pkill -TERM -P "$p" 2>/dev/null || true; kill -TERM "$p" 2>/dev/null || true
                echo "sent SIGTERM to $l $task / $method (pid $p)"
            fi
        done < <(lane_jobs "$l")
    done
}

cmd_check() {   # offline: no LLM calls
    local ok=1
    lane_env
    command -v g++ >/dev/null || { echo "FAIL g++ missing"; ok=0; }
    [[ -f "$FRONTIER_CS_DIR/algorithmic/problems/0/chk.cc" ]] \
        || { echo "FAIL no Frontier-CS checkout at $FRONTIER_CS_DIR (run scripts/server/setup_coral.sh)"; exit 1; }
    "$VENV/bin/python" -c 'import numpy, litellm, levi, skydiscover' 2>/dev/null || { echo "FAIL $VENV cannot import numpy/litellm/levi/skydiscover"; ok=0; }
    [[ -f .env ]] || { echo "FAIL no .env with OPENAI_API_KEY=sk-or-v1-..."; ok=0; }
    echo "== polyominoes: judge the seed (expect compiled, score 0) and Frontier-CS's two example solutions"
    py - <<'EOF' || ok=0
import os, sys
sys.path.insert(0, "benchmarks/coral_tasks/polyominoes")
import judge
ex = judge.PROBLEM_DIR / "examples"
for name, path in [("seed", "benchmarks/coral_tasks/polyominoes/initial_program.cpp"),
                   ("examples/gpt5.cpp", ex / "gpt5.cpp"), ("examples/reference.cpp", ex / "reference.cpp")]:
    if not os.path.exists(path):
        print(f"  {name}: missing"); continue
    r = judge.judge(open(path).read())
    print(f"  {name:24s} {r.feedback(max_lines=0)}  ({r.seconds:.0f}s)")
EOF
    echo "== polyominoes: SpecEvo wrapper on the same seed (expect an 'error': nothing valid)"
    (cd levi && py -c '
import sys; sys.path.insert(0, "examples/polyominoes"); import problem as P
ns = {}; exec(P.SEED_PROGRAM, ns); r = P.score_fn(ns["solution"]); print("  ", {k: r[k] for k in list(r)[:3]})') || ok=0
    echo "== kernel_builder: baseline evaluator and SpecEvo score_fn on the seed (expect correct, same cycles)"
    py -c '
import sys, time; sys.path.insert(0, "benchmarks/coral_tasks/kernel_builder"); import evaluator
t = time.time(); r = evaluator.evaluate("benchmarks/coral_tasks/kernel_builder/initial_program.py")
m = getattr(r, "metrics", r); print("   baseline evaluator:", {k: round(v, 3) for k, v in m.items()}, f"{time.time()-t:.0f}s")' || ok=0
    (cd levi && py -c '
import sys, time; sys.path.insert(0, "examples/kernel_builder"); import problem as K
ns = {}; exec(K.SEED_PROGRAM, ns); t = time.time(); print("   SpecEvo score_fn:  ", K.score_fn(ns["kernel_builder"]), f"{time.time()-t:.0f}s")') || ok=0
    echo "== commands (dry run)"
    DRY_RUN=1 bash "$SELF" start 2>&1 | grep -E "start |ERROR" | sed 's/^/  /' || ok=0
    [[ "$ok" == 1 ]] && echo "check: OK" || { echo "check: FAILED (see above)"; return 1; }
}

cmd_start() {
    local lanes l s env_args k
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    command -v tmux >/dev/null || die "tmux is not installed"
    [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV"
    [[ -f .env ]] || die "no .env with OPENAI_API_KEY=sk-or-v1-..."
    [[ -f "$FRONTIER_CS_DIR/algorithmic/problems/0/chk.cc" ]] || die "no Frontier-CS at $FRONTIER_CS_DIR (scripts/server/setup_coral.sh)"

    if [[ "$DRY_RUN" == 1 ]]; then
        for l in "${lanes[@]}"; do lane_main "$l"; done
        return 0
    fi
    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux has-session -t "=$s" 2>/dev/null && die "tmux session '$s' already exists (status: $0 status, stop: $0 stop $l)"
    done

    local key
    key="$(sed -n 's/^\(OPENROUTER_API_KEY\|OPENAI_API_KEY\)=//p' .env | head -1)"
    if [[ -n "$key" ]] && command -v curl >/dev/null; then
        curl -s -m 15 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $key" | "$VENV/bin/python" -c '
import json, sys
try:
    d = json.load(sys.stdin)["data"]
except Exception:
    print("key check  : could not read /api/v1/key"); sys.exit(0)
lim, rem = d.get("limit"), d.get("limit_remaining")
print("key check  : used $%.2f, limit %s, remaining %s" % (d.get("usage", 0),
      "none" if lim is None else "$%.0f" % lim, "unlimited" if rem is None else "$%.2f" % rem))
' || true
    fi

    env_args="HOME='$HOME'"
    for k in "${KNOBS[@]}"; do env_args+=" $k='${!k}'"; done
    for l in "${lanes[@]}"; do
        mkdir -p "$(lane_dir "$l")"
        tmux new-session -d -s "$(lane_session "$l")" -c "$REPO_ROOT" \
            "env $env_args bash '$SELF' _lane $l 2>&1 | tee -a '$(lane_dir "$l")/lane.log'; \
echo; bash '$SELF' status $l; exec bash"
        printf '%-5s tmux %-10s %-15s %s  ->  %s/\n' "$l" "$(lane_session "$l")" "$(lane_task "$l")" \
            "${METHODS[*]}" "$(lane_dir "$l")"
    done
    cat <<EOF

budget     : each job stops at ${SECONDS_CAP}s or \$${DOLLARS}; up to $(( ${#lanes[@]} * ${#METHODS[@]} )) jobs = \$$(awk "BEGIN{print ${#lanes[@]} * ${#METHODS[@]} * $DOLLARS}") at most
watch      : tmux attach -t $(lane_session "${lanes[0]}")     (detach: Ctrl-b then d)
progress   : $0 status            (live: watch -n 60 $0 status)
table      : $0 compare
stop       : $0 stop [lane ...]
EOF
}

case "${1:-status}" in
    check)   shift; cmd_check ;;
    start)   shift; cmd_start "$@" ;;
    status)  shift; cmd_status "$@" ;;
    compare|table) shift; cmd_compare "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    _lane)   lane_main "${2:?lane}" ;;
    -h|--help) sed -n '2,/^# -\{20,\}$/p' "$0" ;;
    *) die "unknown command '$1' (check | start | status | compare | stop)" ;;
esac
