#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ablation_queue.sh — the four SpecEvo ablations on four tasks, 1h30 each,
# spread over two tmux sessions ("lanes") that each work through their own
# queue one job at a time. Two runs are in flight at any moment.
#
#   ./scripts/server/run_ablation_queue.sh start     # launch both lanes, detached
#   ./scripts/server/run_ablation_queue.sh status    # what is done / running / left
#   ./scripts/server/run_ablation_queue.sh stop      # kill both lanes and their runs
#   DRY_RUN=1 ./scripts/server/run_ablation_queue.sh start   # print all 16 commands, no API calls
#
# Ablations (paper name -> run_bench.sh flag):
#   w/o AST features        -> --ablation emb_only        (behaviour vector = embedding only)
#   Single operator prompt  -> {"single_prompt_operators":true}   (A6)
#   w/o Navigator           -> --ablation no_paradigm     (pe_interval = 0)
#   w/o Advisor             -> --ablation no_meta_advice  (A4)
#
# Tasks:
#   llm_sql      levi/examples/ADRS/llm_sql
#   txn          levi/examples/ADRS/txn_scheduling
#   assortment   levi/examples/co_bench/assortment       CO-Bench (10 files, 5 instances)
#   bin_packing  levi/examples/co_bench/bin_packing_1d   CO-Bench (10 files, 3 instances)
#
# Lanes: one ADRS run and one CO-Bench run are always paired, so the box
# carries one evaluation-heavy CO-Bench run at a time instead of two. Each
# lane goes ablation by ablation, so after every ~3h20 one more ablation has
# all four tasks finished.
#
#   lane 1 (abl_lane1): llm_sql, txn            x  4 ablations = 8 jobs
#   lane 2 (abl_lane2): assortment, bin_packing x  4 ablations = 8 jobs
#
# Restarting is safe: a job whose summary.json exists is skipped; a job that
# died half-way is moved aside to <dir>.incomplete_<ts> and run again.
#
# Knobs (environment variables):
#   GROUP=specevo_ablation   output group -> outputs/server/ablation/$GROUP/<task>__<ablation>/
#   SECONDS_BUDGET=5400      SpecEvo's own graceful wall-clock budget (1h30)
#   HARD_TIMEOUT=6300        hard kill (1h45): graceful stop + in-flight evals finishing
#   DOLLARS=30               per-run spend ceiling, a runaway guard only: 16 x $30 = $480
#                            stays under the key's remaining budget, so the queue cannot
#                            drain the key half-way. Set DOLLARS= (empty) to remove it.
#   EVAL_TIMEOUT=600         per-candidate evaluation timeout (blade.yml default)
#   WORKERS=4                concurrent LLM workers per run
#   PE_INTERVAL=10           Navigator cadence for the arms that keep it (blade.yml default)
#   THREADS_PER_EVAL=2       BLAS/OpenMP threads per evaluator process
#   MUTATION_MODEL / PARADIGM_MODEL   default qwen3-30b-a3b-instruct-2507 / gpt-5
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GROUP="${GROUP:-specevo_ablation}"
GROUP_DIR="outputs/server/ablation/$GROUP"
SECONDS_BUDGET="${SECONDS_BUDGET:-5400}"
HARD_TIMEOUT="${HARD_TIMEOUT:-6300}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"
DOLLARS="${DOLLARS-30}"
WORKERS="${WORKERS:-4}"
PE_INTERVAL="${PE_INTERVAL:-10}"
THREADS_PER_EVAL="${THREADS_PER_EVAL:-2}"
MUTATION_MODEL="${MUTATION_MODEL:-openrouter/qwen/qwen3-30b-a3b-instruct-2507}"
PARADIGM_MODEL="${PARADIGM_MODEL:-openrouter/openai/gpt-5}"
DRY_RUN="${DRY_RUN:-0}"
# run_bench.sh --dry-run still creates the output dir and run.log; keep those
# out of the real group so a later `start` does not mistake them for crashes.
[[ "$DRY_RUN" == 1 ]] && GROUP_DIR="outputs/server/ablation/${GROUP}_dryrun"
VENV="$REPO_ROOT/.venv"

ABLATIONS=(emb_only single_prompt no_paradigm no_meta_advice)
LANE_SESSIONS=(abl_lane1 abl_lane2)
LANE1_TASKS=(llm_sql txn)
LANE2_TASKS=(assortment bin_packing)

die() { echo "ERROR: $*" >&2; exit 1; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }

task_args() {   # task -> run_bench.sh benchmark flags
    case "$1" in
        llm_sql)     echo "--benchmark levi/examples/ADRS/llm_sql" ;;
        txn)         echo "--benchmark levi/examples/ADRS/txn_scheduling" ;;
        assortment)  echo "--benchmark levi/examples/co_bench/assortment --cobench-max-cases 10 --cobench-max-instances 5" ;;
        bin_packing) echo "--benchmark levi/examples/co_bench/bin_packing_1d --cobench-max-cases 10 --cobench-max-instances 3" ;;
        *) die "unknown task '$1'" ;;
    esac
}

lane_tasks() { if [[ "$1" == 1 ]]; then echo "${LANE1_TASKS[*]}"; else echo "${LANE2_TASKS[*]}"; fi; }

# Every job, in lane order: "<lane> <task> <ablation>".
all_jobs() {
    local lane abl task
    for lane in 1 2; do
        for abl in "${ABLATIONS[@]}"; do
            for task in $(lane_tasks "$lane"); do echo "$lane $task $abl"; done
        done
    done
}

job_dir() { echo "$GROUP_DIR/$1__$2"; }

pid_alive() { [[ -s "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

job_state() {   # task ablation -> DONE | RUNNING | TIMEOUT | FAILED | PENDING
    local d; d="$(job_dir "$1" "$2")"
    if   [[ -f "$d/summary.json" ]];  then echo DONE
    elif pid_alive "$d/.run.pid";     then echo RUNNING
    elif [[ -f "$d/.timed_out" ]];    then echo TIMEOUT
    elif [[ -d "$d" ]];               then echo FAILED
    else                                   echo PENDING
    fi
}

# ---------------------------------------------------------------------------
run_job() {     # task ablation
    local task="$1" abl="$2" out abl_args=() rc
    out="$(job_dir "$task" "$abl")"
    case "$abl" in
        emb_only|no_paradigm|no_meta_advice) abl_args=(--ablation "$abl") ;;
        single_prompt) abl_args=(--advanced-options '{"single_prompt_operators":true}') ;;
        *) die "unknown ablation '$abl'" ;;
    esac

    if [[ -f "$out/summary.json" ]]; then
        echo "[$(ts)] skip   $task / $abl (summary.json already there)"
        return 0
    fi
    if [[ -d "$out" && "$DRY_RUN" != 1 ]]; then
        local aside="$out.incomplete_$(date +%Y%m%d-%H%M%S)"
        mv "$out" "$aside"
        echo "[$(ts)] moved  half-finished $out -> $aside"
    fi

    echo "[$(ts)] start  $task / $abl -> $out"
    set +e
    # shellcheck disable=SC2046  # task_args is deliberately word-split
    "$REPO_ROOT/scripts/server/run_bench.sh" blade --no-conda --no-install-deps \
        $(task_args "$task") \
        "${abl_args[@]}" \
        --seconds "$SECONDS_BUDGET" --timeout "$HARD_TIMEOUT" --dollars "$DOLLARS" \
        --mutation-model "$MUTATION_MODEL" --paradigm-model "$PARADIGM_MODEL" \
        --workers "$WORKERS" --pe-interval "$PE_INTERVAL" --eval-timeout "$EVAL_TIMEOUT" \
        --save-eval-code \
        --run-id "${GROUP}_${task}_${abl}" --output-dir "$out" \
        $([[ "$DRY_RUN" == 1 ]] && echo --dry-run)
    rc=$?
    set -e
    echo "[$(ts)] end    $task / $abl  exit=$rc  state=$(job_state "$task" "$abl")"
    return 0    # one failed job must not stop the rest of the lane
}

lane_main() {   # lane number
    local lane="$1" task abl
    # The tmux server may predate this venv, so pin the interpreter here.
    export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
    # Two runs x 4 evaluator processes on 8 cores / 16 threads: cap the
    # numeric libraries so candidates cannot oversubscribe the box.
    export OMP_NUM_THREADS="$THREADS_PER_EVAL" OPENBLAS_NUM_THREADS="$THREADS_PER_EVAL" \
           MKL_NUM_THREADS="$THREADS_PER_EVAL" NUMEXPR_NUM_THREADS="$THREADS_PER_EVAL"
    mkdir -p "$GROUP_DIR"
    echo "[$(ts)] lane $lane begins: tasks=($(lane_tasks "$lane")) ablations=(${ABLATIONS[*]})"
    for abl in "${ABLATIONS[@]}"; do
        for task in $(lane_tasks "$lane"); do
            run_job "$task" "$abl"
        done
    done
    echo "[$(ts)] lane $lane finished."
}

# ---------------------------------------------------------------------------
cmd_status() {
    local py="$VENV/bin/python"; [[ -x "$py" ]] || py=python3
    printf '%-5s %-12s %-15s %-8s %10s %8s %8s %8s\n' lane task ablation state best_dev test cost evals
    local lane task abl st d
    while read -r lane task abl; do
        st="$(job_state "$task" "$abl")"; d="$(job_dir "$task" "$abl")"
        if [[ "$st" == DONE ]]; then
            "$py" - "$lane" "$task" "$abl" "$d/summary.json" <<'PY'
import json, sys
lane, task, abl, path = sys.argv[1:]
s = json.load(open(path)); m = s.get("best_metrics") or {}
f = lambda v: "-" if v is None else f"{float(v):.4f}"
print(f"{lane:<5} {task:<12} {abl:<15} {'DONE':<8} {f(s.get('best_score')):>10} "
      f"{f(m.get('test_score')):>8} {'$'+format(s.get('total_cost', 0), '.2f'):>8} {s.get('total_evaluations', '-'):>8}")
PY
        else
            printf '%-5s %-12s %-15s %-8s\n' "$lane" "$task" "$abl" "$st"
        fi
    done < <(all_jobs)
    echo
    for s in "${LANE_SESSIONS[@]}"; do
        tmux has-session -t "$s" 2>/dev/null && echo "tmux $s: alive" || echo "tmux $s: not running"
    done
}

cmd_stop() {
    local s task abl lane d p
    for s in "${LANE_SESSIONS[@]}"; do tmux kill-session -t "$s" 2>/dev/null && echo "killed tmux $s" || true; done
    # Belt and braces: any run that outlived its pane.
    while read -r lane task abl; do
        d="$(job_dir "$task" "$abl")"
        if pid_alive "$d/.run.pid"; then
            p="$(cat "$d/.run.pid")"
            pkill -TERM -P "$p" 2>/dev/null || true; kill -TERM "$p" 2>/dev/null || true
            echo "sent SIGTERM to $task / $abl (pid $p)"
        fi
    done < <(all_jobs)
}

cmd_start() {
    command -v tmux >/dev/null || die "tmux is not installed"
    [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV (python3 -m venv .venv && .venv/bin/pip install -e . -r scripts/server/requirements-server.txt)"
    "$VENV/bin/python" -c 'import numpy, scipy, sklearn, litellm, pandas' 2>/dev/null \
        || die "$VENV cannot import the BLADE dependencies (pip install -r scripts/server/requirements-server.txt)"
    [[ -f .env ]] || die "no .env with OPENAI_API_KEY=sk-or-v1-... (see docs/SERVER_GUIDE.md section 4)"

    if [[ "$DRY_RUN" == 1 ]]; then
        lane_main 1; lane_main 2; return 0
    fi

    for s in "${LANE_SESSIONS[@]}"; do
        tmux has-session -t "$s" 2>/dev/null && die "tmux session '$s' already exists (status: $0 status, stop: $0 stop)"
    done

    # Remaining key budget, so a near-empty key is noticed before 24h of runs.
    local key
    key="$(sed -n 's/^\(OPENROUTER_API_KEY\|OPENAI_API_KEY\)=//p' .env | head -1)"
    if [[ -n "$key" ]] && command -v curl >/dev/null; then
        curl -s -m 15 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $key" \
          | "$VENV/bin/python" -c '
import json, sys
try:
    d = json.load(sys.stdin)["data"]
except Exception:
    print("key check  : could not read /api/v1/key"); sys.exit(0)
lim, rem = d.get("limit"), d.get("limit_remaining")
lim_s = "none" if lim is None else "$%.0f" % lim
rem_s = "unlimited" if rem is None else "$%.2f" % rem
print("key check  : used $%.2f, limit %s, remaining %s, resets %s"
      % (d.get("usage", 0), lim_s, rem_s, d.get("limit_reset") or "never"))
' || true
    fi

    mkdir -p "$GROUP_DIR"
    local lane log
    for lane in 1 2; do
        log="$GROUP_DIR/lane$lane.log"
        tmux new-session -d -s "${LANE_SESSIONS[$((lane-1))]}" -c "$REPO_ROOT" \
            "env GROUP='$GROUP' SECONDS_BUDGET='$SECONDS_BUDGET' HARD_TIMEOUT='$HARD_TIMEOUT' \
EVAL_TIMEOUT='$EVAL_TIMEOUT' DOLLARS='$DOLLARS' WORKERS='$WORKERS' PE_INTERVAL='$PE_INTERVAL' THREADS_PER_EVAL='$THREADS_PER_EVAL' \
MUTATION_MODEL='$MUTATION_MODEL' PARADIGM_MODEL='$PARADIGM_MODEL' HOME='$HOME' \
bash '$REPO_ROOT/scripts/server/run_ablation_queue.sh' _lane $lane 2>&1 | tee -a '$log'; \
echo; bash '$REPO_ROOT/scripts/server/run_ablation_queue.sh' status; exec bash"
    done

    cat <<EOF
group      : $GROUP  ->  $GROUP_DIR/
per job    : ${SECONDS_BUDGET}s budget (hard kill at ${HARD_TIMEOUT}s), cap \$${DOLLARS:-none}, $WORKERS LLM workers, eval timeout ${EVAL_TIMEOUT}s, navigator every $PE_INTERVAL evals
models     : $MUTATION_MODEL  +  $PARADIGM_MODEL
lane 1     : tmux ${LANE_SESSIONS[0]}  ->  ${LANE1_TASKS[*]}  x  ${ABLATIONS[*]}
lane 2     : tmux ${LANE_SESSIONS[1]}  ->  ${LANE2_TASKS[*]}  x  ${ABLATIONS[*]}

watch      : tmux attach -t ${LANE_SESSIONS[0]}     (detach: Ctrl-b then d)
progress   : $0 status
lane logs  : tail -f $GROUP_DIR/lane1.log $GROUP_DIR/lane2.log
stop all   : $0 stop
EOF
}

case "${1:-start}" in
    start)  cmd_start ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    _lane)  lane_main "${2:?lane number}" ;;
    -h|--help) sed -n '2,45p' "$0" ;;
    *) die "unknown command '$1' (start | status | stop)" ;;
esac
